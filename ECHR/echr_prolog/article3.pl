violation(article3, Person) :-
    state_responsible_for_mistreatment(Person),
    (
        torture(Person)
    ;   inhuman_treatment(Person)
    ;   degrading_treatment(Person)
    ).