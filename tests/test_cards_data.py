"""Lightweight, dependency-free validation for Phase 0 card declarations."""
import copy
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "data" / "cards.schema.json").read_text(encoding="utf-8"))
DATA = json.loads((ROOT / "data" / "cards.json").read_text(encoding="utf-8"))


class ValidationError(ValueError):
    pass


def resolve(schema):
    while "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            raise AssertionError("unsupported reference: " + ref)
        node = SCHEMA
        for part in ref[2:].split("/"):
            node = node[part]
        schema = node
    return schema


def error(path, value, message):
    raise ValidationError("%s: %s (value=%r)" % (path, message, value))


def type_matches(value, expected):
    expected = expected if isinstance(expected, list) else [expected]
    checks = {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "boolean": lambda: isinstance(value, bool),
        "null": lambda: value is None,
    }
    return any(checks[name]() for name in expected)


def validate(value, schema, path="$"):
    schema = resolve(schema)
    if "anyOf" in schema:
        messages = []
        for candidate in schema["anyOf"]:
            try:
                validate(value, candidate, path)
                break
            except ValidationError as exc:
                messages.append(str(exc))
        else:
            error(path, value, "does not match any permitted form: " + "; ".join(messages))
    if "const" in schema and value != schema["const"]:
        error(path, value, "must equal %r" % schema["const"])
    if "enum" in schema and value not in schema["enum"]:
        error(path, value, "must be one of %r" % schema["enum"])
    if "type" in schema and not type_matches(value, schema["type"]):
        error(path, value, "must have type %r" % schema["type"])
    if isinstance(value, str) and "pattern" in schema and not re.fullmatch(schema["pattern"], value):
        error(path, value, "does not match pattern %s" % schema["pattern"])
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            error(path, value, "must be >= %r" % schema["minimum"])
        if "maximum" in schema and value > schema["maximum"]:
            error(path, value, "must be <= %r" % schema["maximum"])
    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                error(path + "/" + key, None, "required field is missing")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    error(path + "/" + key, value[key], "unknown property")
        for key, child in properties.items():
            if key in value:
                validate(value[key], child, path + "/" + key)
        property_names = schema.get("propertyNames")
        if property_names:
            for key in value:
                validate(key, property_names, path + "/" + key)
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            error(path, value, "must contain at least %d items" % schema["minItems"])
        if schema.get("uniqueItems") and len({json.dumps(item, sort_keys=True, ensure_ascii=False) for item in value}) != len(value):
            error(path, value, "items must be unique")
        if "items" in schema:
            for index, item in enumerate(value):
                validate(item, schema["items"], path + "/" + str(index))
    for condition in schema.get("allOf", []):
        if "if" in condition and "then" in condition:
            try:
                validate(value, condition["if"], path)
            except ValidationError:
                continue
            validate(value, condition["then"], path)


# hook -> (timing, exactly-required params keys)
HOOK_SPEC = {
    "on_place__gain_consumer_goods": ("on_place", {"amount"}),
    "on_place__slash_and_burn": ("on_place", {"amount"}),
    "on_place__gain_from_household": ("on_place", {"amount"}),
    "on_place__design_office": ("on_place", {"amount"}),
    "on_place__discard_then_draw": ("on_place", {"discard", "draw"}),
    "on_place__orchard": ("on_place", {"target_total"}),
    "on_place__build": ("on_place", {"discount"}),
    "on_place__build_farm_free": ("on_place", set()),
    "on_place__build_then_draw": ("on_place", {"draw"}),
    "on_place__build_two_same_cost": ("on_place", set()),
    "on_place__draw_cards": ("on_place", {"amount"}),
    "on_place__chemical_plant": ("on_place", set()),
    "on_place__discard_gain_household": ("on_place", {"discard", "amount"}),
    "on_place__quarry": ("on_place", {"draw"}),
    "on_place__hire_worker": ("on_place", {"amount"}),
    "on_place__hire_worker_up_to": ("on_place", {"target_total"}),
    "on_place__hire_worker_ready": ("on_place", {"amount"}),
    "on_round_start__slash_and_burn_collect": ("on_round_start", set()),
    "passive__hand_limit_bonus": ("passive", {"amount"}),
    "passive__worker_cap_bonus": ("passive", {"amount"}),
    "passive__none": ("passive", set()),
    "passive__consumer_goods": ("passive", set()),
    "on_game_end__forgive_unpaid_wages": ("on_game_end", {"amount"}),
    "on_game_end__score_per_building": ("on_game_end", {"points_per"}),
    "on_game_end__score_per_consumer_good": ("on_game_end", {"points_per"}),
    "on_game_end__score_per_worker": ("on_game_end", {"points_per"}),
    "on_game_end__score_per_locked_building": ("on_game_end", {"points_per"}),
    "on_game_end__score_per_factory": ("on_game_end", {"points_per"}),
    "on_game_end__unpaid_wage": ("on_game_end", {"points_per"}),
}

# Transcription pin against RULES.md chapter 11:
# card_id -> (card_type, copies, build_cost, printed_value, ((effect_id, hook, params), ...))
EXPECTED_CARDS = {
    "base:farm": ("building", 8, 1, 6, (("EFF-farm-01", "on_place__gain_consumer_goods", {"amount": 2}),)),
    "base:slash-and-burn": ("building", 2, 1, 0, (("EFF-slash-and-burn-01", "on_place__slash_and_burn", {"amount": 5}), ("EFF-slash-and-burn-02", "on_round_start__slash_and_burn_collect", {}))),
    "base:cafe": ("building", 2, 1, 8, (("EFF-cafe-01", "on_place__gain_from_household", {"amount": 5}),)),
    "base:design-office": ("building", 4, 1, 8, (("EFF-design-office-01", "on_place__design_office", {"amount": 5}),)),
    "base:factory": ("building", 8, 2, 12, (("EFF-factory-01", "on_place__discard_then_draw", {"discard": 2, "draw": 4}),)),
    "base:orchard": ("building", 3, 2, 10, (("EFF-orchard-01", "on_place__orchard", {"target_total": 4}),)),
    "base:construction-company": ("building", 4, 2, 10, (("EFF-construction-company-01", "on_place__build", {"discount": 1}),)),
    "base:warehouse": ("building", 3, 2, 10, (("EFF-warehouse-01", "passive__hand_limit_bonus", {"amount": 4}),)),
    "base:company-housing": ("building", 2, 2, 8, (("EFF-company-housing-01", "passive__worker_cap_bonus", {"amount": 1}),)),
    "base:law-office": ("building", 1, 2, 8, (("EFF-law-office-01", "on_game_end__forgive_unpaid_wages", {"amount": 5}),)),
    "base:plantation": ("building", 3, 3, 12, (("EFF-plantation-01", "on_place__gain_consumer_goods", {"amount": 3}),)),
    "base:restaurant": ("building", 2, 3, 16, (("EFF-restaurant-01", "on_place__discard_gain_household", {"discard": 1, "amount": 15}),)),
    "base:pioneer": ("building", 2, 3, 14, (("EFF-pioneer-01", "on_place__build_farm_free", {}),)),
    "base:realtor": ("building", 2, 3, 10, (("EFF-realtor-01", "on_game_end__score_per_building", {"points_per": 3}),)),
    "base:agricultural-coop": ("building", 1, 3, 12, (("EFF-agricultural-coop-01", "on_game_end__score_per_consumer_good", {"points_per": 3}),)),
    "base:general-contractor": ("building", 3, 4, 18, (("EFF-general-contractor-01", "on_place__build_then_draw", {"draw": 2}),)),
    "base:steel-mill": ("building", 3, 4, 20, (("EFF-steel-mill-01", "on_place__draw_cards", {"amount": 3}),)),
    "base:chemical-plant": ("building", 2, 4, 18, (("EFF-chemical-plant-01", "on_place__chemical_plant", {}),)),
    "base:mansion": ("building", 1, 4, 28, (("EFF-mansion-01", "passive__none", {}),)),
    "base:labor-union": ("building", 1, 4, 0, (("EFF-labor-union-01", "on_game_end__score_per_worker", {"points_per": 6}),)),
    "base:automobile-factory": ("building", 3, 5, 24, (("EFF-automobile-factory-01", "on_place__discard_then_draw", {"discard": 3, "draw": 7}),)),
    "base:two-for-one": ("building", 2, 5, 20, (("EFF-two-for-one-01", "on_place__build_two_same_cost", {}),)),
    "base:headquarters": ("building", 1, 5, 20, (("EFF-headquarters-01", "on_game_end__score_per_locked_building", {"points_per": 6}),)),
    "base:railroad": ("building", 1, 5, 18, (("EFF-railroad-01", "on_game_end__score_per_factory", {"points_per": 8}),)),
    "base:quarry": ("public_workplace", 1, None, None, (("EFF-quarry-01", "on_place__quarry", {"draw": 1}),)),
    "base:mine": ("public_workplace", 1, None, None, (("EFF-mine-01", "on_place__draw_cards", {"amount": 1}),)),
    "base:school": ("public_workplace", 1, None, None, (("EFF-school-01", "on_place__hire_worker", {"amount": 1}),)),
    "base:carpenter": ("public_workplace", 3, None, None, (("EFF-carpenter-01", "on_place__build", {"discount": 0}),)),
    "base:consumer-goods": ("consumer_goods", 24, None, None, (("EFF-consumer-goods-01", "passive__consumer_goods", {}),)),
    "base:unpaid-wage": ("unpaid_wage", 15, None, None, (("EFF-unpaid-wage-01", "on_game_end__unpaid_wage", {"points_per": 3}),)),
    "base:round-1": ("round", 1, None, None, (("EFF-stall-01", "on_place__discard_gain_household", {"discard": 1, "amount": 6}),)),
    "base:round-2": ("round", 1, None, None, (("EFF-market-01", "on_place__discard_gain_household", {"discard": 2, "amount": 12}),)),
    "base:round-3": ("round", 1, None, None, (("EFF-high-school-01", "on_place__hire_worker_up_to", {"target_total": 4}),)),
    "base:round-4": ("round", 1, None, None, (("EFF-supermarket-01", "on_place__discard_gain_household", {"discard": 3, "amount": 18}),)),
    "base:round-5": ("round", 1, None, None, (("EFF-university-01", "on_place__hire_worker_up_to", {"target_total": 5}),)),
    "base:round-6": ("round", 1, None, None, (("EFF-department-store-01", "on_place__discard_gain_household", {"discard": 4, "amount": 24}),)),
    "base:round-7": ("round", 1, None, None, (("EFF-vocational-school-01", "on_place__hire_worker_ready", {"amount": 1}),)),
    "base:round-8": ("round", 1, None, None, (("EFF-expo-01", "on_place__discard_gain_household", {"discard": 5, "amount": 30}),)),
    "base:round-9": ("round", 1, None, None, ()),
}

# RUL-WAGE-001
WAGE_TABLE = {1: 2, 2: 2, 3: 3, 4: 3, 5: 3, 6: 4, 7: 4, 8: 5, 9: 5}


def semantic_validate(data):
    cards = data["cards"]
    ids = [card["card_id"] for card in cards]
    if len(ids) != len(set(ids)):
        raise ValidationError("$/cards: duplicate card_id (value=%r)" % ids)
    effects = [effect["effect_id"] for card in cards for effect in card["effects"]]
    if len(effects) != len(set(effects)):
        raise ValidationError("$/cards: duplicate effect_id (value=%r)" % effects)
    rules_text = (ROOT / "RULES.md").read_text(encoding="utf-8")
    # source_refs が指す調査ノート(docs/phase0-sources.md)は本リポジトリに含めていないため照合しない。
    for index, card in enumerate(cards):
        label = "$/cards/%d[%s]" % (index, card["card_id"])
        if card["status"] == "provisional" and not card.get("open_issue_refs"):
            raise ValidationError("%s/open_issue_refs: provisional card has no issue reference (value=%r)" % (label, card.get("open_issue_refs")))
        for field, text in (("rules_refs", rules_text), ("open_issue_refs", rules_text)):
            for ref in card.get(field, []):
                if ref not in text:
                    raise ValidationError("%s/%s: reference is absent from source document (value=%r)" % (label, field, ref))
        for effect in card["effects"]:
            if effect["effect_id"] not in rules_text:
                raise ValidationError("%s/effects: effect is absent from RULES.md (value=%r)" % (label, effect["effect_id"]))
            spec = HOOK_SPEC.get(effect["hook"])
            if spec is None:
                raise ValidationError("%s/effects: hook has no HOOK_SPEC entry (value=%r)" % (label, effect["hook"]))
            if effect["timing"] != spec[0]:
                raise ValidationError("%s/effects/%s: timing must be %r for this hook (value=%r)" % (label, effect["effect_id"], spec[0], effect["timing"]))
            if set(effect["params"].keys()) != spec[1]:
                raise ValidationError("%s/effects/%s: params keys must be exactly %r (value=%r)" % (label, effect["effect_id"], sorted(spec[1]), effect["params"]))
            if effect["mandatory"] is not True:
                raise ValidationError("%s/effects/%s: all Phase 0 effects are mandatory (value=%r)" % (label, effect["effect_id"], effect["mandatory"]))
        # cross-field structural constraints
        has_workplace_tag = "workplace" in card["tags"]
        if has_workplace_tag != (card["workplace"] is not None):
            raise ValidationError("%s: workplace tag and workplace object must agree (value=%r)" % (label, card["tags"]))
        if "facility" in card["tags"] and (has_workplace_tag or card["workplace"] is not None):
            raise ValidationError("%s: facility card must not be a workplace (value=%r)" % (label, card["tags"]))
        if card["workplace"] is not None:
            occ = card["workplace"]["occupancy"]
            if occ["mode"] == "exclusive" and not isinstance(occ["capacity"], int):
                raise ValidationError("%s: exclusive occupancy needs integer capacity (value=%r)" % (label, occ))
            if occ["mode"] == "unlimited" and occ["capacity"] is not None:
                raise ValidationError("%s: unlimited occupancy needs null capacity (value=%r)" % (label, occ))
        if card["card_type"] == "building":
            if not isinstance(card["build_cost"], int) or not isinstance(card["printed_value"], int):
                raise ValidationError("%s: building must have integer build_cost and printed_value (value=%r/%r)" % (label, card["build_cost"], card["printed_value"]))
        else:
            if card["build_cost"] is not None or card["printed_value"] is not None:
                raise ValidationError("%s: non-building must have null build_cost and printed_value (value=%r/%r)" % (label, card["build_cost"], card["printed_value"]))
        if card["card_type"] == "round":
            info = card.get("round")
            if info is None:
                raise ValidationError("%s/round: round card needs round info (value=%r)" % (label, info))
            n = info["round_number"]
            if info["wage_rate"] != WAGE_TABLE[n]:
                raise ValidationError("%s/round: wage_rate must follow RUL-WAGE-001 (value=%r; expected=%r)" % (label, info["wage_rate"], WAGE_TABLE[n]))
            expected_face = n + 1 if n <= 8 else None
            if info["public_face_from_round"] != expected_face:
                raise ValidationError("%s/round: public_face_from_round must be %r (value=%r)" % (label, expected_face, info["public_face_from_round"]))
            if expected_face is None and (card["workplace"] is not None or card["effects"]):
                raise ValidationError("%s: round card without public face must have no workplace and no effects (value=%r)" % (label, card["card_id"]))
            if expected_face is not None and (card["workplace"] is None or not card["effects"]):
                raise ValidationError("%s: round card with public face needs workplace and effects (value=%r)" % (label, card["card_id"]))
        elif "round" in card:
            raise ValidationError("%s/round: only round cards may carry round info (value=%r)" % (label, card["round"]))
    # transcription pin against RULES.md chapter 11
    actual = {}
    for card in cards:
        actual[card["card_id"]] = (
            card["card_type"], card["copies"], card["build_cost"], card["printed_value"],
            tuple((effect["effect_id"], effect["hook"], effect["params"]) for effect in card["effects"]),
        )
    if set(actual) != set(EXPECTED_CARDS):
        raise ValidationError("$/cards: card set differs from RULES.md (missing=%r; extra=%r)" % (sorted(set(EXPECTED_CARDS) - set(actual)), sorted(set(actual) - set(EXPECTED_CARDS))))
    for card_id, expected in EXPECTED_CARDS.items():
        if actual[card_id] != expected:
            raise ValidationError("$/cards[%s]: transcription differs from RULES.md (value=%r; expected=%r)" % (card_id, actual[card_id], expected))
    totals = {}
    for card in cards:
        totals[card["card_type"]] = totals.get(card["card_type"], 0) + card["copies"]
    expected_totals = {"building": 64, "public_workplace": 6, "round": 9, "consumer_goods": 24, "unpaid_wage": 15}
    if totals != expected_totals:
        raise ValidationError("$/cards: card copy totals differ (value=%r; expected=%r)" % (totals, expected_totals))


def full_validate(data):
    validate(data, SCHEMA)
    semantic_validate(data)


def card_by_id(data, card_id):
    for card in data["cards"]:
        if card["card_id"] == card_id:
            return card
    raise AssertionError("card not found: " + card_id)


class CardsDataTests(unittest.TestCase):
    def test_valid_data_passes_structure_and_semantics(self):
        full_validate(DATA)

    def test_each_mutation_is_rejected(self):
        cases = [
            ("unknown top-level key", lambda d: d.__setitem__("unknown", True)),
            ("unknown card field", lambda d: d["cards"][0].__setitem__("unknown", True)),
            ("unknown tag", lambda d: d["cards"][0]["tags"].append("unknown_tag")),
            ("unknown card_type", lambda d: d["cards"][0].__setitem__("card_type", "unknown_type")),
            ("unknown timing", lambda d: d["cards"][0]["effects"][0].__setitem__("timing", "unknown_timing")),
            ("unknown hook", lambda d: d["cards"][0]["effects"][0].__setitem__("hook", "unknown_hook")),
            ("unknown params key", lambda d: d["cards"][0]["effects"][0]["params"].__setitem__("unknown", 1)),
            ("negative copies", lambda d: d["cards"][0].__setitem__("copies", -1)),
            ("invalid card_id", lambda d: d["cards"][0].__setitem__("card_id", "bad id")),
            ("missing required field", lambda d: d["cards"][0].pop("name")),
            ("provisional without issue", lambda d: card_by_id(d, "base:slash-and-burn").pop("open_issue_refs")),
            ("round_number above maximum", lambda d: card_by_id(d, "base:round-1")["round"].__setitem__("round_number", 10)),
            ("wage_rate above maximum", lambda d: card_by_id(d, "base:round-1")["round"].__setitem__("wage_rate", 6)),
            ("public_face_from_round above maximum", lambda d: card_by_id(d, "base:round-1")["round"].__setitem__("public_face_from_round", 10)),
            ("timing not matching hook", lambda d: card_by_id(d, "base:farm")["effects"][0].__setitem__("timing", "passive")),
            ("missing required param for hook", lambda d: card_by_id(d, "base:farm")["effects"][0].__setitem__("params", {})),
            ("param not allowed for hook", lambda d: card_by_id(d, "base:farm")["effects"][0].__setitem__("params", {"amount": 2, "points_per": 1})),
            ("optional effect", lambda d: card_by_id(d, "base:farm")["effects"][0].__setitem__("mandatory", False)),
            ("workplace tag without workplace object", lambda d: card_by_id(d, "base:farm").__setitem__("workplace", None)),
            ("facility with workplace object", lambda d: card_by_id(d, "base:warehouse").__setitem__("workplace", {"occupancy": {"mode": "exclusive", "capacity": 1}, "usable_by": "owner_only", "owner_benefit_hook": None})),
            ("unlimited occupancy with capacity", lambda d: card_by_id(d, "base:mine")["workplace"]["occupancy"].__setitem__("capacity", 1)),
            ("building with null build_cost", lambda d: card_by_id(d, "base:farm").__setitem__("build_cost", None)),
            ("non-building with build_cost", lambda d: card_by_id(d, "base:quarry").__setitem__("build_cost", 1)),
            ("non-round card with round info", lambda d: card_by_id(d, "base:farm").__setitem__("round", {"round_number": 1, "wage_rate": 2, "public_face_from_round": 2})),
            ("wage table mismatch", lambda d: card_by_id(d, "base:round-3")["round"].__setitem__("wage_rate", 2)),
            ("face round mismatch", lambda d: card_by_id(d, "base:round-1")["round"].__setitem__("public_face_from_round", 3)),
            ("faceless round card with workplace", lambda d: card_by_id(d, "base:round-9").__setitem__("tags", ["workplace"]) or card_by_id(d, "base:round-9").__setitem__("workplace", {"occupancy": {"mode": "exclusive", "capacity": 1}, "usable_by": "any_player", "owner_benefit_hook": None})),
            ("effect_id swap between cards", lambda d: (
                card_by_id(d, "base:farm")["effects"][0].__setitem__("effect_id", "EFF-plantation-01"),
                card_by_id(d, "base:farm")["rules_refs"].__setitem__(0, "EFF-plantation-01"),
                card_by_id(d, "base:plantation")["effects"][0].__setitem__("effect_id", "EFF-farm-01"),
                card_by_id(d, "base:plantation")["rules_refs"].__setitem__(0, "EFF-farm-01"),
            )),
            ("transcription drift", lambda d: card_by_id(d, "base:farm")["effects"][0]["params"].__setitem__("amount", 3)),
            ("copies drift", lambda d: card_by_id(d, "base:farm").__setitem__("copies", 7)),
        ]
        for name, mutate in cases:
            with self.subTest(name=name):
                value = copy.deepcopy(DATA)
                mutate(value)
                with self.assertRaises(ValidationError) as raised:
                    full_validate(value)
                message = str(raised.exception)
                self.assertIn("value=", message)
                self.assertIn("$/", message)


if __name__ == "__main__":
    unittest.main()
