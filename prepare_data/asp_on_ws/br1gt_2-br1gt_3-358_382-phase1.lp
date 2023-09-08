
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

	orientation(br1gt_2,169,358). coords(br1gt_2,86,207,358).  visible(br1gt_2,358). happensAt(inactive(br1gt_2),358). orientation(br1gt_3,162,358). coords(br1gt_3,30,163,358).  visible(br1gt_3,358). happensAt(walking(br1gt_3),358).
orientation(br1gt_2,169,359). coords(br1gt_2,86,207,359).  visible(br1gt_2,359). happensAt(inactive(br1gt_2),359). orientation(br1gt_3,162,359). coords(br1gt_3,30,163,359).  visible(br1gt_3,359). happensAt(walking(br1gt_3),359).
orientation(br1gt_2,169,360). coords(br1gt_2,86,207,360).  visible(br1gt_2,360). happensAt(inactive(br1gt_2),360). orientation(br1gt_3,162,360). coords(br1gt_3,31,163,360).  visible(br1gt_3,360). happensAt(walking(br1gt_3),360).
orientation(br1gt_2,169,361). coords(br1gt_2,86,207,361).  visible(br1gt_2,361). happensAt(inactive(br1gt_2),361). orientation(br1gt_3,162,361). coords(br1gt_3,32,163,361).  visible(br1gt_3,361). happensAt(walking(br1gt_3),361).
orientation(br1gt_2,169,362). coords(br1gt_2,86,207,362).  visible(br1gt_2,362). happensAt(inactive(br1gt_2),362). orientation(br1gt_3,162,362). coords(br1gt_3,32,161,362).  visible(br1gt_3,362). happensAt(walking(br1gt_3),362).
orientation(br1gt_2,169,363). coords(br1gt_2,86,207,363).  visible(br1gt_2,363). happensAt(inactive(br1gt_2),363). orientation(br1gt_3,162,363). coords(br1gt_3,32,161,363).  visible(br1gt_3,363). happensAt(walking(br1gt_3),363).
orientation(br1gt_2,169,364). coords(br1gt_2,86,207,364).  visible(br1gt_2,364). happensAt(inactive(br1gt_2),364). orientation(br1gt_3,162,364). coords(br1gt_3,32,161,364).  visible(br1gt_3,364). happensAt(walking(br1gt_3),364).
orientation(br1gt_2,169,365). coords(br1gt_2,86,207,365).  visible(br1gt_2,365). happensAt(inactive(br1gt_2),365). orientation(br1gt_3,162,365). coords(br1gt_3,32,161,365).  visible(br1gt_3,365). happensAt(walking(br1gt_3),365).
orientation(br1gt_2,169,366). coords(br1gt_2,86,207,366).  visible(br1gt_2,366). happensAt(inactive(br1gt_2),366). orientation(br1gt_3,162,366). coords(br1gt_3,32,161,366).  visible(br1gt_3,366). happensAt(walking(br1gt_3),366).
orientation(br1gt_2,169,367). coords(br1gt_2,86,207,367).  visible(br1gt_2,367). happensAt(inactive(br1gt_2),367). orientation(br1gt_3,162,367). coords(br1gt_3,32,161,367).  visible(br1gt_3,367). happensAt(walking(br1gt_3),367).
orientation(br1gt_2,169,368). coords(br1gt_2,86,207,368).  visible(br1gt_2,368). happensAt(inactive(br1gt_2),368). orientation(br1gt_3,162,368). coords(br1gt_3,33,159,368).  visible(br1gt_3,368). happensAt(walking(br1gt_3),368).
orientation(br1gt_2,169,369). coords(br1gt_2,86,207,369).  visible(br1gt_2,369). happensAt(inactive(br1gt_2),369). orientation(br1gt_3,162,369). coords(br1gt_3,33,159,369).  visible(br1gt_3,369). happensAt(walking(br1gt_3),369).
orientation(br1gt_2,169,370). coords(br1gt_2,86,207,370).  visible(br1gt_2,370). happensAt(inactive(br1gt_2),370). orientation(br1gt_3,162,370). coords(br1gt_3,34,159,370).  visible(br1gt_3,370). happensAt(walking(br1gt_3),370).
orientation(br1gt_2,169,371). coords(br1gt_2,86,207,371).  visible(br1gt_2,371). happensAt(inactive(br1gt_2),371). orientation(br1gt_3,162,371). coords(br1gt_3,34,159,371).  visible(br1gt_3,371). happensAt(walking(br1gt_3),371).
orientation(br1gt_2,169,372). coords(br1gt_2,86,207,372).  visible(br1gt_2,372). happensAt(inactive(br1gt_2),372). orientation(br1gt_3,162,372). coords(br1gt_3,34,159,372).  visible(br1gt_3,372). happensAt(walking(br1gt_3),372).
orientation(br1gt_2,169,373). coords(br1gt_2,86,207,373).  visible(br1gt_2,373). happensAt(inactive(br1gt_2),373). orientation(br1gt_3,162,373). coords(br1gt_3,34,159,373).  visible(br1gt_3,373). happensAt(walking(br1gt_3),373).
orientation(br1gt_2,169,374). coords(br1gt_2,86,207,374).  visible(br1gt_2,374). happensAt(inactive(br1gt_2),374). orientation(br1gt_3,162,374). coords(br1gt_3,35,159,374).  visible(br1gt_3,374). happensAt(walking(br1gt_3),374).
orientation(br1gt_2,169,375). coords(br1gt_2,86,208,375).  visible(br1gt_2,375). happensAt(inactive(br1gt_2),375). orientation(br1gt_3,162,375). coords(br1gt_3,35,159,375).  visible(br1gt_3,375). happensAt(walking(br1gt_3),375).
orientation(br1gt_2,169,376). coords(br1gt_2,86,208,376).  visible(br1gt_2,376). happensAt(inactive(br1gt_2),376). orientation(br1gt_3,162,376). coords(br1gt_3,35,159,376).  visible(br1gt_3,376). happensAt(walking(br1gt_3),376).
orientation(br1gt_2,169,377). coords(br1gt_2,86,208,377).  visible(br1gt_2,377). happensAt(inactive(br1gt_2),377). orientation(br1gt_3,162,377). coords(br1gt_3,35,159,377).  visible(br1gt_3,377). happensAt(walking(br1gt_3),377).
orientation(br1gt_2,169,378). coords(br1gt_2,86,208,378).  visible(br1gt_2,378). happensAt(inactive(br1gt_2),378). orientation(br1gt_3,162,378). coords(br1gt_3,35,159,378).  visible(br1gt_3,378). happensAt(walking(br1gt_3),378).
orientation(br1gt_2,169,379). coords(br1gt_2,86,208,379).  visible(br1gt_2,379). happensAt(inactive(br1gt_2),379). orientation(br1gt_3,162,379). coords(br1gt_3,35,159,379).  visible(br1gt_3,379). happensAt(walking(br1gt_3),379).
orientation(br1gt_2,169,380). coords(br1gt_2,86,208,380).  visible(br1gt_2,380). happensAt(inactive(br1gt_2),380). orientation(br1gt_3,162,380). coords(br1gt_3,35,159,380).  visible(br1gt_3,380). happensAt(walking(br1gt_3),380).
orientation(br1gt_2,169,381). coords(br1gt_2,86,208,381).  visible(br1gt_2,381). happensAt(inactive(br1gt_2),381). orientation(br1gt_3,162,381). coords(br1gt_3,35,159,381).  visible(br1gt_3,381). happensAt(walking(br1gt_3),381).
orientation(br1gt_2,169,382). coords(br1gt_2,86,208,382).  visible(br1gt_2,382). happensAt(inactive(br1gt_2),382). orientation(br1gt_3,162,382). coords(br1gt_3,37,159,382).  visible(br1gt_3,382). happensAt(walking(br1gt_3),382).
#show holdsAt/2.