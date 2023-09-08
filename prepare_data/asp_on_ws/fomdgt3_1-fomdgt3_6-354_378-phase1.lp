
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

	orientation(fomdgt3_1,150,354). coords(fomdgt3_1,54,100,354).  visible(fomdgt3_1,354). happensAt(inactive(fomdgt3_1),354). orientation(fomdgt3_6,150,354). coords(fomdgt3_6,193,100,354).  visible(fomdgt3_6,354). happensAt(walking(fomdgt3_6),354).
orientation(fomdgt3_1,150,355). coords(fomdgt3_1,54,100,355).  visible(fomdgt3_1,355). happensAt(inactive(fomdgt3_1),355). orientation(fomdgt3_6,150,355). coords(fomdgt3_6,193,93,355).  visible(fomdgt3_6,355). happensAt(walking(fomdgt3_6),355).
orientation(fomdgt3_1,150,356). coords(fomdgt3_1,54,100,356).  visible(fomdgt3_1,356). happensAt(inactive(fomdgt3_1),356). orientation(fomdgt3_6,150,356). coords(fomdgt3_6,186,90,356).  visible(fomdgt3_6,356). happensAt(walking(fomdgt3_6),356).
orientation(fomdgt3_1,150,357). coords(fomdgt3_1,54,100,357).  visible(fomdgt3_1,357). happensAt(inactive(fomdgt3_1),357). orientation(fomdgt3_6,156,357). coords(fomdgt3_6,184,90,357).  visible(fomdgt3_6,357). happensAt(walking(fomdgt3_6),357).
orientation(fomdgt3_1,150,358). coords(fomdgt3_1,54,100,358).  visible(fomdgt3_1,358). happensAt(inactive(fomdgt3_1),358). orientation(fomdgt3_6,156,358). coords(fomdgt3_6,182,90,358).  visible(fomdgt3_6,358). happensAt(walking(fomdgt3_6),358).
orientation(fomdgt3_1,150,359). coords(fomdgt3_1,46,102,359).  visible(fomdgt3_1,359). happensAt(inactive(fomdgt3_1),359). orientation(fomdgt3_6,156,359). coords(fomdgt3_6,181,90,359).  visible(fomdgt3_6,359). happensAt(walking(fomdgt3_6),359).
orientation(fomdgt3_1,150,360). coords(fomdgt3_1,46,102,360).  visible(fomdgt3_1,360). happensAt(inactive(fomdgt3_1),360). orientation(fomdgt3_6,156,360). coords(fomdgt3_6,178,94,360).  visible(fomdgt3_6,360). happensAt(walking(fomdgt3_6),360).
orientation(fomdgt3_1,150,361). coords(fomdgt3_1,46,102,361).  visible(fomdgt3_1,361). happensAt(inactive(fomdgt3_1),361). orientation(fomdgt3_6,156,361). coords(fomdgt3_6,176,94,361).  visible(fomdgt3_6,361). happensAt(walking(fomdgt3_6),361).
orientation(fomdgt3_1,150,362). coords(fomdgt3_1,46,102,362).  visible(fomdgt3_1,362). happensAt(inactive(fomdgt3_1),362). orientation(fomdgt3_6,156,362). coords(fomdgt3_6,170,95,362).  visible(fomdgt3_6,362). happensAt(walking(fomdgt3_6),362).
orientation(fomdgt3_1,150,363). coords(fomdgt3_1,46,102,363).  visible(fomdgt3_1,363). happensAt(inactive(fomdgt3_1),363). orientation(fomdgt3_6,156,363). coords(fomdgt3_6,170,95,363).  visible(fomdgt3_6,363). happensAt(walking(fomdgt3_6),363).
orientation(fomdgt3_1,150,364). coords(fomdgt3_1,46,102,364).  visible(fomdgt3_1,364). happensAt(inactive(fomdgt3_1),364). orientation(fomdgt3_6,156,364). coords(fomdgt3_6,159,95,364).  visible(fomdgt3_6,364). happensAt(running(fomdgt3_6),364).
orientation(fomdgt3_1,150,365). coords(fomdgt3_1,46,102,365).  visible(fomdgt3_1,365). happensAt(inactive(fomdgt3_1),365). orientation(fomdgt3_6,156,365). coords(fomdgt3_6,157,95,365).  visible(fomdgt3_6,365). happensAt(running(fomdgt3_6),365).
orientation(fomdgt3_1,150,366). coords(fomdgt3_1,46,102,366).  visible(fomdgt3_1,366). happensAt(inactive(fomdgt3_1),366). orientation(fomdgt3_6,156,366). coords(fomdgt3_6,155,94,366).  visible(fomdgt3_6,366). happensAt(running(fomdgt3_6),366).
orientation(fomdgt3_1,150,367). coords(fomdgt3_1,46,102,367).  visible(fomdgt3_1,367). happensAt(inactive(fomdgt3_1),367). orientation(fomdgt3_6,156,367). coords(fomdgt3_6,152,94,367).  visible(fomdgt3_6,367). happensAt(running(fomdgt3_6),367).
orientation(fomdgt3_1,150,368). coords(fomdgt3_1,46,102,368).  visible(fomdgt3_1,368). happensAt(inactive(fomdgt3_1),368). orientation(fomdgt3_6,156,368). coords(fomdgt3_6,148,93,368).  visible(fomdgt3_6,368). happensAt(running(fomdgt3_6),368).
orientation(fomdgt3_1,150,369). coords(fomdgt3_1,46,102,369).  visible(fomdgt3_1,369). happensAt(inactive(fomdgt3_1),369). orientation(fomdgt3_6,156,369). coords(fomdgt3_6,147,93,369).  visible(fomdgt3_6,369). happensAt(running(fomdgt3_6),369).
orientation(fomdgt3_1,150,370). coords(fomdgt3_1,46,102,370).  visible(fomdgt3_1,370). happensAt(inactive(fomdgt3_1),370). orientation(fomdgt3_6,156,370). coords(fomdgt3_6,140,94,370).  visible(fomdgt3_6,370). happensAt(running(fomdgt3_6),370).
orientation(fomdgt3_1,150,371). coords(fomdgt3_1,46,102,371).  visible(fomdgt3_1,371). happensAt(inactive(fomdgt3_1),371). orientation(fomdgt3_6,156,371). coords(fomdgt3_6,136,94,371).  visible(fomdgt3_6,371). happensAt(running(fomdgt3_6),371).
orientation(fomdgt3_1,150,372). coords(fomdgt3_1,46,102,372).  visible(fomdgt3_1,372). happensAt(inactive(fomdgt3_1),372). orientation(fomdgt3_6,156,372). coords(fomdgt3_6,134,94,372).  visible(fomdgt3_6,372). happensAt(running(fomdgt3_6),372).
orientation(fomdgt3_1,150,373). coords(fomdgt3_1,46,102,373).  visible(fomdgt3_1,373). happensAt(inactive(fomdgt3_1),373). orientation(fomdgt3_6,156,373). coords(fomdgt3_6,132,93,373).  visible(fomdgt3_6,373). happensAt(running(fomdgt3_6),373).
orientation(fomdgt3_1,150,374). coords(fomdgt3_1,46,102,374).  visible(fomdgt3_1,374). happensAt(inactive(fomdgt3_1),374). orientation(fomdgt3_6,156,374). coords(fomdgt3_6,130,92,374).  visible(fomdgt3_6,374). happensAt(running(fomdgt3_6),374).
orientation(fomdgt3_1,150,375). coords(fomdgt3_1,46,102,375).  visible(fomdgt3_1,375). happensAt(inactive(fomdgt3_1),375). orientation(fomdgt3_6,156,375). coords(fomdgt3_6,127,91,375).  visible(fomdgt3_6,375). happensAt(running(fomdgt3_6),375).
orientation(fomdgt3_1,150,376). coords(fomdgt3_1,46,102,376).  visible(fomdgt3_1,376). happensAt(inactive(fomdgt3_1),376). orientation(fomdgt3_6,156,376). coords(fomdgt3_6,124,91,376).  visible(fomdgt3_6,376). happensAt(running(fomdgt3_6),376).
orientation(fomdgt3_1,150,377). coords(fomdgt3_1,46,102,377).  visible(fomdgt3_1,377). happensAt(inactive(fomdgt3_1),377). orientation(fomdgt3_6,156,377). coords(fomdgt3_6,120,91,377).  visible(fomdgt3_6,377). happensAt(running(fomdgt3_6),377).
orientation(fomdgt3_1,150,378). coords(fomdgt3_1,46,102,378).  visible(fomdgt3_1,378). happensAt(inactive(fomdgt3_1),378). orientation(fomdgt3_6,156,378). coords(fomdgt3_6,118,91,378).  visible(fomdgt3_6,378). happensAt(running(fomdgt3_6),378).
#show holdsAt/2.