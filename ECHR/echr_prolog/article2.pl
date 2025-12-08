violation(article2, Person) :-
    state_responsible_for_death(Person),
    deprivation_of_life(Person),
    \+ judicial_execution(Person),
    \+ justified_force(Person).

deprivation_of_life(Person) :-
    life_taken_intentionally(Person);  
    death_resulted_from_force(Person).

justified_force(Person) :-
    force_used_against(Person),
    force_absolutely_necessary(Person),
    (   defence_from_unlawful_violence(Person)
    ;   effect_lawful_arrest(Person)
    ;   prevent_escape_of_lawfully_detained_individual(Person)
    ;   quell_riot_or_insurrection(Person)
    ).

judicial_execution(Person) :-
    convicted_of(Person, Crime),
    death_penalty_provided_by_law(Crime).