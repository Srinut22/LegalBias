violation(article6_1, Person) :-
    trial(Person),
    (   \+ independent_tribunal(Person)
    ;   \+ impartial_tribunal(Person)
    ;   \+ tribunal_established_by_law(Person)
    ;   \+ hearing_within_reasonable_time(Person)
    ;   \+ fair_hearing(Person)
    ;   \+ public_judgment_or_justified_exclusion(Person)
    ).

violation(article6_2, Person) :-
    charged(Person, _Offence),
    \+ presumed_innocent(Person).

violation(article6_3, Person) :-
    charged(Person, _Offence),
    (   \+ informed_promptly_in_language_understood_of_nature_and_cause(Person)
    ;   \+ adequate_time_and_facilities_for_defence(Person)
    ;   \+ effective_legal_assistance(Person)
    ;   (   relevant_witness_evidence(Person),
            \+ opportunity_to_examine_witnesses_on_equal_terms(Person)
        )
    ;   (   needs_interpreter(Person),
            \+ interpreter_provided_free_of_charge(Person)
        )
    ).

violation(article6, Person) :-
    violation(article6_1, Person).
violation(article6, Person) :-
    violation(article6_2, Person).
violation(article6, Person) :-
    violation(article6_3, Person).
