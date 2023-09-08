
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

	orientation(fomdgt3_1,150,360). coords(fomdgt3_1,46,102,360).  visible(fomdgt3_1,360). happensAt(inactive(fomdgt3_1),360). orientation(fomdgt3_2,42,360). coords(fomdgt3_2,262,93,360).  visible(fomdgt3_2,360). happensAt(walking(fomdgt3_2),360).
orientation(fomdgt3_1,150,361). coords(fomdgt3_1,46,102,361).  visible(fomdgt3_1,361). happensAt(inactive(fomdgt3_1),361). orientation(fomdgt3_2,42,361). coords(fomdgt3_2,262,93,361).  visible(fomdgt3_2,361). happensAt(walking(fomdgt3_2),361).
orientation(fomdgt3_1,150,362). coords(fomdgt3_1,46,102,362).  visible(fomdgt3_1,362). happensAt(inactive(fomdgt3_1),362). orientation(fomdgt3_2,42,362). coords(fomdgt3_2,262,95,362).  visible(fomdgt3_2,362). happensAt(walking(fomdgt3_2),362).
orientation(fomdgt3_1,150,363). coords(fomdgt3_1,46,102,363).  visible(fomdgt3_1,363). happensAt(inactive(fomdgt3_1),363). orientation(fomdgt3_2,42,363). coords(fomdgt3_2,262,95,363).  visible(fomdgt3_2,363). happensAt(walking(fomdgt3_2),363).
orientation(fomdgt3_1,150,364). coords(fomdgt3_1,46,102,364).  visible(fomdgt3_1,364). happensAt(inactive(fomdgt3_1),364). orientation(fomdgt3_2,42,364). coords(fomdgt3_2,262,95,364).  visible(fomdgt3_2,364). happensAt(inactive(fomdgt3_2),364).
orientation(fomdgt3_1,150,365). coords(fomdgt3_1,46,102,365).  visible(fomdgt3_1,365). happensAt(inactive(fomdgt3_1),365). orientation(fomdgt3_2,42,365). coords(fomdgt3_2,262,95,365).  visible(fomdgt3_2,365). happensAt(inactive(fomdgt3_2),365).
orientation(fomdgt3_1,150,366). coords(fomdgt3_1,46,102,366).  visible(fomdgt3_1,366). happensAt(inactive(fomdgt3_1),366). orientation(fomdgt3_2,42,366). coords(fomdgt3_2,262,95,366).  visible(fomdgt3_2,366). happensAt(inactive(fomdgt3_2),366).
orientation(fomdgt3_1,150,367). coords(fomdgt3_1,46,102,367).  visible(fomdgt3_1,367). happensAt(inactive(fomdgt3_1),367). orientation(fomdgt3_2,42,367). coords(fomdgt3_2,262,95,367).  visible(fomdgt3_2,367). happensAt(inactive(fomdgt3_2),367).
orientation(fomdgt3_1,150,368). coords(fomdgt3_1,46,102,368).  visible(fomdgt3_1,368). happensAt(inactive(fomdgt3_1),368). orientation(fomdgt3_2,42,368). coords(fomdgt3_2,262,95,368).  visible(fomdgt3_2,368). happensAt(inactive(fomdgt3_2),368).
orientation(fomdgt3_1,150,369). coords(fomdgt3_1,46,102,369).  visible(fomdgt3_1,369). happensAt(inactive(fomdgt3_1),369). orientation(fomdgt3_2,42,369). coords(fomdgt3_2,262,95,369).  visible(fomdgt3_2,369). happensAt(inactive(fomdgt3_2),369).
orientation(fomdgt3_1,150,370). coords(fomdgt3_1,46,102,370).  visible(fomdgt3_1,370). happensAt(inactive(fomdgt3_1),370). orientation(fomdgt3_2,42,370). coords(fomdgt3_2,262,95,370).  visible(fomdgt3_2,370). happensAt(inactive(fomdgt3_2),370).
orientation(fomdgt3_1,150,371). coords(fomdgt3_1,46,102,371).  visible(fomdgt3_1,371). happensAt(inactive(fomdgt3_1),371). orientation(fomdgt3_2,42,371). coords(fomdgt3_2,262,97,371).  visible(fomdgt3_2,371). happensAt(inactive(fomdgt3_2),371).
orientation(fomdgt3_1,150,372). coords(fomdgt3_1,46,102,372).  visible(fomdgt3_1,372). happensAt(inactive(fomdgt3_1),372). orientation(fomdgt3_2,42,372). coords(fomdgt3_2,262,96,372).  visible(fomdgt3_2,372). happensAt(inactive(fomdgt3_2),372).
orientation(fomdgt3_1,150,373). coords(fomdgt3_1,46,102,373).  visible(fomdgt3_1,373). happensAt(inactive(fomdgt3_1),373). orientation(fomdgt3_2,42,373). coords(fomdgt3_2,261,96,373).  visible(fomdgt3_2,373). happensAt(inactive(fomdgt3_2),373).
orientation(fomdgt3_1,150,374). coords(fomdgt3_1,46,102,374).  visible(fomdgt3_1,374). happensAt(inactive(fomdgt3_1),374). orientation(fomdgt3_2,42,374). coords(fomdgt3_2,262,96,374).  visible(fomdgt3_2,374). happensAt(inactive(fomdgt3_2),374).
orientation(fomdgt3_1,150,375). coords(fomdgt3_1,46,102,375).  visible(fomdgt3_1,375). happensAt(inactive(fomdgt3_1),375). orientation(fomdgt3_2,42,375). coords(fomdgt3_2,262,96,375).  visible(fomdgt3_2,375). happensAt(inactive(fomdgt3_2),375).
orientation(fomdgt3_1,150,376). coords(fomdgt3_1,46,102,376).  visible(fomdgt3_1,376). happensAt(inactive(fomdgt3_1),376). orientation(fomdgt3_2,42,376). coords(fomdgt3_2,262,96,376).  visible(fomdgt3_2,376). happensAt(inactive(fomdgt3_2),376).
orientation(fomdgt3_1,150,377). coords(fomdgt3_1,46,102,377).  visible(fomdgt3_1,377). happensAt(inactive(fomdgt3_1),377). orientation(fomdgt3_2,42,377). coords(fomdgt3_2,262,96,377).  visible(fomdgt3_2,377). happensAt(inactive(fomdgt3_2),377).
orientation(fomdgt3_1,150,378). coords(fomdgt3_1,46,102,378).  visible(fomdgt3_1,378). happensAt(inactive(fomdgt3_1),378). orientation(fomdgt3_2,42,378). coords(fomdgt3_2,262,96,378).  visible(fomdgt3_2,378). happensAt(inactive(fomdgt3_2),378).
orientation(fomdgt3_1,150,379). coords(fomdgt3_1,46,102,379).  visible(fomdgt3_1,379). happensAt(inactive(fomdgt3_1),379). orientation(fomdgt3_2,42,379). coords(fomdgt3_2,262,96,379).  visible(fomdgt3_2,379). happensAt(inactive(fomdgt3_2),379).
orientation(fomdgt3_1,150,380). coords(fomdgt3_1,46,102,380).  visible(fomdgt3_1,380). happensAt(inactive(fomdgt3_1),380). orientation(fomdgt3_2,42,380). coords(fomdgt3_2,262,96,380).  visible(fomdgt3_2,380). happensAt(inactive(fomdgt3_2),380).
orientation(fomdgt3_1,150,381). coords(fomdgt3_1,46,102,381).  visible(fomdgt3_1,381). happensAt(inactive(fomdgt3_1),381). orientation(fomdgt3_2,42,381). coords(fomdgt3_2,262,96,381).  visible(fomdgt3_2,381). happensAt(inactive(fomdgt3_2),381).
orientation(fomdgt3_1,150,382). coords(fomdgt3_1,46,102,382).  visible(fomdgt3_1,382). happensAt(inactive(fomdgt3_1),382). orientation(fomdgt3_2,42,382). coords(fomdgt3_2,262,96,382).  visible(fomdgt3_2,382). happensAt(inactive(fomdgt3_2),382).
orientation(fomdgt3_1,150,383). coords(fomdgt3_1,46,102,383).  visible(fomdgt3_1,383). happensAt(inactive(fomdgt3_1),383). orientation(fomdgt3_2,42,383). coords(fomdgt3_2,262,96,383).  visible(fomdgt3_2,383). happensAt(inactive(fomdgt3_2),383).
orientation(fomdgt3_1,150,384). coords(fomdgt3_1,46,102,384).  visible(fomdgt3_1,384). happensAt(inactive(fomdgt3_1),384). orientation(fomdgt3_2,42,384). coords(fomdgt3_2,262,96,384).  visible(fomdgt3_2,384). happensAt(inactive(fomdgt3_2),384).
#show holdsAt/2.