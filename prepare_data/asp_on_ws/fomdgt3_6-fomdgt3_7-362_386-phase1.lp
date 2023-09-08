
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
orientation(fomdgt3_6,156,375). coords(fomdgt3_6,127,91,375).  visible(fomdgt3_6,375). happensAt(running(fomdgt3_6),375). orientation(fomdgt3_7,134,375). coords(fomdgt3_7,78,86,375).  visible(fomdgt3_7,375). happensAt(walking(fomdgt3_7),375).
orientation(fomdgt3_6,156,376). coords(fomdgt3_6,124,91,376).  visible(fomdgt3_6,376). happensAt(running(fomdgt3_6),376). orientation(fomdgt3_7,134,376). coords(fomdgt3_7,77,87,376).  visible(fomdgt3_7,376). happensAt(walking(fomdgt3_7),376).
orientation(fomdgt3_6,156,377). coords(fomdgt3_6,120,91,377).  visible(fomdgt3_6,377). happensAt(running(fomdgt3_6),377). orientation(fomdgt3_7,134,377). coords(fomdgt3_7,77,88,377).  visible(fomdgt3_7,377). happensAt(walking(fomdgt3_7),377).
orientation(fomdgt3_6,156,378). coords(fomdgt3_6,118,91,378).  visible(fomdgt3_6,378). happensAt(running(fomdgt3_6),378). orientation(fomdgt3_7,134,378). coords(fomdgt3_7,77,89,378).  visible(fomdgt3_7,378). happensAt(walking(fomdgt3_7),378).
orientation(fomdgt3_6,156,379). coords(fomdgt3_6,116,91,379).  visible(fomdgt3_6,379). happensAt(running(fomdgt3_6),379). orientation(fomdgt3_7,134,379). coords(fomdgt3_7,77,90,379).  visible(fomdgt3_7,379). happensAt(walking(fomdgt3_7),379).
orientation(fomdgt3_6,156,380). coords(fomdgt3_6,115,91,380).  visible(fomdgt3_6,380). happensAt(running(fomdgt3_6),380). orientation(fomdgt3_7,134,380). coords(fomdgt3_7,77,91,380).  visible(fomdgt3_7,380). happensAt(walking(fomdgt3_7),380).
orientation(fomdgt3_6,156,381). coords(fomdgt3_6,114,90,381).  visible(fomdgt3_6,381). happensAt(running(fomdgt3_6),381). orientation(fomdgt3_7,134,381). coords(fomdgt3_7,77,91,381).  visible(fomdgt3_7,381). happensAt(walking(fomdgt3_7),381).
orientation(fomdgt3_6,156,382). coords(fomdgt3_6,113,90,382).  visible(fomdgt3_6,382). happensAt(running(fomdgt3_6),382). orientation(fomdgt3_7,134,382). coords(fomdgt3_7,77,91,382).  visible(fomdgt3_7,382). happensAt(walking(fomdgt3_7),382).
orientation(fomdgt3_6,156,383). coords(fomdgt3_6,108,89,383).  visible(fomdgt3_6,383). happensAt(running(fomdgt3_6),383). orientation(fomdgt3_7,134,383). coords(fomdgt3_7,77,91,383).  visible(fomdgt3_7,383). happensAt(walking(fomdgt3_7),383).
orientation(fomdgt3_6,140,384). coords(fomdgt3_6,106,89,384).  visible(fomdgt3_6,384). happensAt(running(fomdgt3_6),384). orientation(fomdgt3_7,134,384). coords(fomdgt3_7,77,92,384).  visible(fomdgt3_7,384). happensAt(walking(fomdgt3_7),384).
orientation(fomdgt3_6,140,385). coords(fomdgt3_6,103,88,385).  visible(fomdgt3_6,385). happensAt(running(fomdgt3_6),385). orientation(fomdgt3_7,134,385). coords(fomdgt3_7,77,93,385).  visible(fomdgt3_7,385). happensAt(walking(fomdgt3_7),385).
orientation(fomdgt3_6,140,386). coords(fomdgt3_6,102,88,386).  visible(fomdgt3_6,386). happensAt(running(fomdgt3_6),386). orientation(fomdgt3_7,134,386). coords(fomdgt3_7,77,93,386).  visible(fomdgt3_7,386). happensAt(walking(fomdgt3_7),386).
#show holdsAt/2.