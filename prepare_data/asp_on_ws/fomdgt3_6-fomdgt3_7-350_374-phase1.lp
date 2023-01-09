
holdsAt(F,Te) :- initiatedAt(F,Ts), fluent(F), next(Ts,Te).
holdsAt(F,Te) :- holdsAt(F,Ts), not terminatedAt(F,Ts), fluent(F), next(Ts,Te).


fluent(moving(X,Y)) :- person(X), person(Y), X != Y.
fluent(meeting(X,Y)) :- person(X), person(Y), X != Y.


#include "../next.lp".

% Distance/orientation calculations:
#include "../distances.lp".

% Definitions of "type" predicates (extracted from the data):
time(T) :- happensAt(disappear(_),T).
time(T) :- happensAt(appear(_),T).
time(T) :- happensAt(active(_),T).
time(T) :- happensAt(inactive(_),T).
time(T) :- happensAt(walking(_),T).
time(T) :- happensAt(running(_),T).
time(T) :- coords(_,_,_,T).
time(T) :- orientation(_,_,T).


person(X) :- happensAt(disappear(X),_).
person(X) :- happensAt(appear(X),_).
person(X) :- happensAt(active(X),_).
person(X) :- happensAt(inactive(X),_).
person(X) :- happensAt(walking(X),_).
person(X) :- happensAt(running(X),_).
person(X) :- coords(X,_,_,_).
person(X) :- orientation(X,_,_).


% Definition for the "moving" complex event: 
initiatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),happensAt(walking(X1),X2),orientationMove(X0,X1,X2),close(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,34,X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(active(X1),X2),happensAt(inactive(X0),X2).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(running(X1),X2), person(X0).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(moving(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).

% Definition for the "meeting" complex event:
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(active(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(active(X1),X2),close(X0,X1,25,X2).
initiatedAt(meeting(X0,X1),X2) :- happensAt(inactive(X0),X2),happensAt(inactive(X1),X2),close(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X0),X2),person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(running(X1),X2),person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X0),X2), person(X1).
terminatedAt(meeting(X0,X1),X2) :- happensAt(disappear(X1),X2), person(X0).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X0),X2),far(X0,X1,25,X2).
terminatedAt(meeting(X0,X1),X2) :- happensAt(walking(X1),X2),far(X0,X1,25,X2).

	orientation(fomdgt3_6,150,350). coords(fomdgt3_6,200,103,350).  visible(fomdgt3_6,350). happensAt(walking(fomdgt3_6),350). orientation(fomdgt3_7,134,350). coords(fomdgt3_7,78,71,350).  visible(fomdgt3_7,350). happensAt(walking(fomdgt3_7),350).
orientation(fomdgt3_6,150,351). coords(fomdgt3_6,199,102,351).  visible(fomdgt3_6,351). happensAt(walking(fomdgt3_6),351). orientation(fomdgt3_7,134,351). coords(fomdgt3_7,78,71,351).  visible(fomdgt3_7,351). happensAt(walking(fomdgt3_7),351).
orientation(fomdgt3_6,150,352). coords(fomdgt3_6,196,100,352).  visible(fomdgt3_6,352). happensAt(walking(fomdgt3_6),352). orientation(fomdgt3_7,134,352). coords(fomdgt3_7,78,71,352).  visible(fomdgt3_7,352). happensAt(walking(fomdgt3_7),352).
orientation(fomdgt3_6,150,353). coords(fomdgt3_6,193,100,353).  visible(fomdgt3_6,353). happensAt(walking(fomdgt3_6),353). orientation(fomdgt3_7,134,353). coords(fomdgt3_7,78,75,353).  visible(fomdgt3_7,353). happensAt(walking(fomdgt3_7),353).
orientation(fomdgt3_6,150,354). coords(fomdgt3_6,193,100,354).  visible(fomdgt3_6,354). happensAt(walking(fomdgt3_6),354). orientation(fomdgt3_7,134,354). coords(fomdgt3_7,78,75,354).  visible(fomdgt3_7,354). happensAt(walking(fomdgt3_7),354).
orientation(fomdgt3_6,150,355). coords(fomdgt3_6,193,93,355).  visible(fomdgt3_6,355). happensAt(walking(fomdgt3_6),355). orientation(fomdgt3_7,134,355). coords(fomdgt3_7,78,75,355).  visible(fomdgt3_7,355). happensAt(walking(fomdgt3_7),355).
orientation(fomdgt3_6,150,356). coords(fomdgt3_6,186,90,356).  visible(fomdgt3_6,356). happensAt(walking(fomdgt3_6),356). orientation(fomdgt3_7,134,356). coords(fomdgt3_7,78,76,356).  visible(fomdgt3_7,356). happensAt(walking(fomdgt3_7),356).
orientation(fomdgt3_6,156,357). coords(fomdgt3_6,184,90,357).  visible(fomdgt3_6,357). happensAt(walking(fomdgt3_6),357). orientation(fomdgt3_7,134,357). coords(fomdgt3_7,78,76,357).  visible(fomdgt3_7,357). happensAt(walking(fomdgt3_7),357).
orientation(fomdgt3_6,156,358). coords(fomdgt3_6,182,90,358).  visible(fomdgt3_6,358). happensAt(walking(fomdgt3_6),358). orientation(fomdgt3_7,134,358). coords(fomdgt3_7,78,76,358).  visible(fomdgt3_7,358). happensAt(walking(fomdgt3_7),358).
orientation(fomdgt3_6,156,359). coords(fomdgt3_6,181,90,359).  visible(fomdgt3_6,359). happensAt(walking(fomdgt3_6),359). orientation(fomdgt3_7,134,359). coords(fomdgt3_7,78,76,359).  visible(fomdgt3_7,359). happensAt(walking(fomdgt3_7),359).
orientation(fomdgt3_6,156,360). coords(fomdgt3_6,178,94,360).  visible(fomdgt3_6,360). happensAt(walking(fomdgt3_6),360). orientation(fomdgt3_7,134,360). coords(fomdgt3_7,78,76,360).  visible(fomdgt3_7,360). happensAt(walking(fomdgt3_7),360).
orientation(fomdgt3_6,156,361). coords(fomdgt3_6,176,94,361).  visible(fomdgt3_6,361). happensAt(walking(fomdgt3_6),361). orientation(fomdgt3_7,134,361). coords(fomdgt3_7,78,76,361).  visible(fomdgt3_7,361). happensAt(walking(fomdgt3_7),361).
orientation(fomdgt3_6,156,362). coords(fomdgt3_6,170,95,362).  visible(fomdgt3_6,362). happensAt(walking(fomdgt3_6),362). orientation(fomdgt3_7,134,362). coords(fomdgt3_7,78,76,362).  visible(fomdgt3_7,362). happensAt(walking(fomdgt3_7),362).
orientation(fomdgt3_6,156,363). coords(fomdgt3_6,170,95,363).  visible(fomdgt3_6,363). happensAt(walking(fomdgt3_6),363). orientation(fomdgt3_7,134,363). coords(fomdgt3_7,78,79,363).  visible(fomdgt3_7,363). happensAt(walking(fomdgt3_7),363).
orientation(fomdgt3_6,156,364). coords(fomdgt3_6,159,95,364).  visible(fomdgt3_6,364). happensAt(running(fomdgt3_6),364). orientation(fomdgt3_7,134,364). coords(fomdgt3_7,78,81,364).  visible(fomdgt3_7,364). happensAt(walking(fomdgt3_7),364).
orientation(fomdgt3_6,156,365). coords(fomdgt3_6,157,95,365).  visible(fomdgt3_6,365). happensAt(running(fomdgt3_6),365). orientation(fomdgt3_7,134,365). coords(fomdgt3_7,78,81,365).  visible(fomdgt3_7,365). happensAt(walking(fomdgt3_7),365).
orientation(fomdgt3_6,156,366). coords(fomdgt3_6,155,94,366).  visible(fomdgt3_6,366). happensAt(running(fomdgt3_6),366). orientation(fomdgt3_7,134,366). coords(fomdgt3_7,78,82,366).  visible(fomdgt3_7,366). happensAt(walking(fomdgt3_7),366).
orientation(fomdgt3_6,156,367). coords(fomdgt3_6,152,94,367).  visible(fomdgt3_6,367). happensAt(running(fomdgt3_6),367). orientation(fomdgt3_7,134,367). coords(fomdgt3_7,78,83,367).  visible(fomdgt3_7,367). happensAt(walking(fomdgt3_7),367).
orientation(fomdgt3_6,156,368). coords(fomdgt3_6,148,93,368).  visible(fomdgt3_6,368). happensAt(running(fomdgt3_6),368). orientation(fomdgt3_7,134,368). coords(fomdgt3_7,78,84,368).  visible(fomdgt3_7,368). happensAt(walking(fomdgt3_7),368).
orientation(fomdgt3_6,156,369). coords(fomdgt3_6,147,93,369).  visible(fomdgt3_6,369). happensAt(running(fomdgt3_6),369). orientation(fomdgt3_7,134,369). coords(fomdgt3_7,78,85,369).  visible(fomdgt3_7,369). happensAt(walking(fomdgt3_7),369).
orientation(fomdgt3_6,156,370). coords(fomdgt3_6,140,94,370).  visible(fomdgt3_6,370). happensAt(running(fomdgt3_6),370). orientation(fomdgt3_7,134,370). coords(fomdgt3_7,78,84,370).  visible(fomdgt3_7,370). happensAt(walking(fomdgt3_7),370).
orientation(fomdgt3_6,156,371). coords(fomdgt3_6,136,94,371).  visible(fomdgt3_6,371). happensAt(running(fomdgt3_6),371). orientation(fomdgt3_7,134,371). coords(fomdgt3_7,78,84,371).  visible(fomdgt3_7,371). happensAt(walking(fomdgt3_7),371).
orientation(fomdgt3_6,156,372). coords(fomdgt3_6,134,94,372).  visible(fomdgt3_6,372). happensAt(running(fomdgt3_6),372). orientation(fomdgt3_7,134,372). coords(fomdgt3_7,78,85,372).  visible(fomdgt3_7,372). happensAt(walking(fomdgt3_7),372).
orientation(fomdgt3_6,156,373). coords(fomdgt3_6,132,93,373).  visible(fomdgt3_6,373). happensAt(running(fomdgt3_6),373). orientation(fomdgt3_7,134,373). coords(fomdgt3_7,78,85,373).  visible(fomdgt3_7,373). happensAt(walking(fomdgt3_7),373).
orientation(fomdgt3_6,156,374). coords(fomdgt3_6,130,92,374).  visible(fomdgt3_6,374). happensAt(running(fomdgt3_6),374). orientation(fomdgt3_7,134,374). coords(fomdgt3_7,78,85,374).  visible(fomdgt3_7,374). happensAt(walking(fomdgt3_7),374).
#show holdsAt/2.