
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

	orientation(bww1gt_1,4,362). coords(bww1gt_1,105,264,362).  visible(bww1gt_1,362). happensAt(inactive(bww1gt_1),362). orientation(bww1gt_2,156,362). coords(bww1gt_2,87,155,362).  visible(bww1gt_2,362). happensAt(walking(bww1gt_2),362).
orientation(bww1gt_1,3,363). coords(bww1gt_1,105,264,363).  visible(bww1gt_1,363). happensAt(inactive(bww1gt_1),363). orientation(bww1gt_2,158,363). coords(bww1gt_2,87,154,363).  visible(bww1gt_2,363). happensAt(walking(bww1gt_2),363).
orientation(bww1gt_1,7,364). coords(bww1gt_1,105,264,364).  visible(bww1gt_1,364). happensAt(inactive(bww1gt_1),364). orientation(bww1gt_2,151,364). coords(bww1gt_2,87,154,364).  visible(bww1gt_2,364). happensAt(walking(bww1gt_2),364).
orientation(bww1gt_1,177,365). coords(bww1gt_1,105,264,365).  visible(bww1gt_1,365). happensAt(inactive(bww1gt_1),365). orientation(bww1gt_2,156,365). coords(bww1gt_2,87,154,365).  visible(bww1gt_2,365). happensAt(walking(bww1gt_2),365).
orientation(bww1gt_1,179,366). coords(bww1gt_1,105,264,366).  visible(bww1gt_1,366). happensAt(inactive(bww1gt_1),366). orientation(bww1gt_2,157,366). coords(bww1gt_2,88,154,366).  visible(bww1gt_2,366). happensAt(walking(bww1gt_2),366).
orientation(bww1gt_1,1,367). coords(bww1gt_1,105,264,367).  visible(bww1gt_1,367). happensAt(inactive(bww1gt_1),367). orientation(bww1gt_2,159,367). coords(bww1gt_2,89,153,367).  visible(bww1gt_2,367). happensAt(walking(bww1gt_2),367).
orientation(bww1gt_1,3,368). coords(bww1gt_1,105,264,368).  visible(bww1gt_1,368). happensAt(inactive(bww1gt_1),368). orientation(bww1gt_2,157,368). coords(bww1gt_2,90,152,368).  visible(bww1gt_2,368). happensAt(walking(bww1gt_2),368).
orientation(bww1gt_1,4,369). coords(bww1gt_1,105,264,369).  visible(bww1gt_1,369). happensAt(inactive(bww1gt_1),369). orientation(bww1gt_2,159,369). coords(bww1gt_2,92,152,369).  visible(bww1gt_2,369). happensAt(walking(bww1gt_2),369).
orientation(bww1gt_1,4,370). coords(bww1gt_1,105,264,370).  visible(bww1gt_1,370). happensAt(inactive(bww1gt_1),370). orientation(bww1gt_2,159,370). coords(bww1gt_2,93,152,370).  visible(bww1gt_2,370). happensAt(walking(bww1gt_2),370).
orientation(bww1gt_1,3,371). coords(bww1gt_1,105,264,371).  visible(bww1gt_1,371). happensAt(inactive(bww1gt_1),371). orientation(bww1gt_2,160,371). coords(bww1gt_2,93,152,371).  visible(bww1gt_2,371). happensAt(walking(bww1gt_2),371).
orientation(bww1gt_1,3,372). coords(bww1gt_1,105,264,372).  visible(bww1gt_1,372). happensAt(inactive(bww1gt_1),372). orientation(bww1gt_2,159,372). coords(bww1gt_2,94,152,372).  visible(bww1gt_2,372). happensAt(walking(bww1gt_2),372).
orientation(bww1gt_1,6,373). coords(bww1gt_1,105,264,373).  visible(bww1gt_1,373). happensAt(inactive(bww1gt_1),373). orientation(bww1gt_2,158,373). coords(bww1gt_2,95,152,373).  visible(bww1gt_2,373). happensAt(walking(bww1gt_2),373).
orientation(bww1gt_1,0,374). coords(bww1gt_1,105,264,374).  visible(bww1gt_1,374). happensAt(inactive(bww1gt_1),374). orientation(bww1gt_2,160,374). coords(bww1gt_2,95,152,374).  visible(bww1gt_2,374). happensAt(walking(bww1gt_2),374).
orientation(bww1gt_1,1,375). coords(bww1gt_1,105,264,375).  visible(bww1gt_1,375). happensAt(inactive(bww1gt_1),375). orientation(bww1gt_2,162,375). coords(bww1gt_2,96,152,375).  visible(bww1gt_2,375). happensAt(walking(bww1gt_2),375).
orientation(bww1gt_1,3,376). coords(bww1gt_1,105,264,376).  visible(bww1gt_1,376). happensAt(inactive(bww1gt_1),376). orientation(bww1gt_2,161,376). coords(bww1gt_2,96,152,376).  visible(bww1gt_2,376). happensAt(walking(bww1gt_2),376).
orientation(bww1gt_1,4,377). coords(bww1gt_1,105,264,377).  visible(bww1gt_1,377). happensAt(inactive(bww1gt_1),377). orientation(bww1gt_2,159,377). coords(bww1gt_2,96,151,377).  visible(bww1gt_2,377). happensAt(walking(bww1gt_2),377).
orientation(bww1gt_1,2,378). coords(bww1gt_1,105,264,378).  visible(bww1gt_1,378). happensAt(inactive(bww1gt_1),378). orientation(bww1gt_2,157,378). coords(bww1gt_2,96,151,378).  visible(bww1gt_2,378). happensAt(walking(bww1gt_2),378).
orientation(bww1gt_1,3,379). coords(bww1gt_1,105,264,379).  visible(bww1gt_1,379). happensAt(inactive(bww1gt_1),379). orientation(bww1gt_2,158,379). coords(bww1gt_2,96,151,379).  visible(bww1gt_2,379). happensAt(walking(bww1gt_2),379).
orientation(bww1gt_1,5,380). coords(bww1gt_1,105,264,380).  visible(bww1gt_1,380). happensAt(inactive(bww1gt_1),380). orientation(bww1gt_2,159,380). coords(bww1gt_2,97,151,380).  visible(bww1gt_2,380). happensAt(walking(bww1gt_2),380).
orientation(bww1gt_1,4,381). coords(bww1gt_1,105,264,381).  visible(bww1gt_1,381). happensAt(inactive(bww1gt_1),381). orientation(bww1gt_2,157,381). coords(bww1gt_2,97,151,381).  visible(bww1gt_2,381). happensAt(walking(bww1gt_2),381).
orientation(bww1gt_1,1,382). coords(bww1gt_1,105,264,382).  visible(bww1gt_1,382). happensAt(inactive(bww1gt_1),382). orientation(bww1gt_2,158,382). coords(bww1gt_2,98,150,382).  visible(bww1gt_2,382). happensAt(walking(bww1gt_2),382).
orientation(bww1gt_1,178,383). coords(bww1gt_1,105,264,383).  visible(bww1gt_1,383). happensAt(inactive(bww1gt_1),383). orientation(bww1gt_2,163,383). coords(bww1gt_2,101,150,383).  visible(bww1gt_2,383). happensAt(walking(bww1gt_2),383).
orientation(bww1gt_1,179,384). coords(bww1gt_1,105,264,384).  visible(bww1gt_1,384). happensAt(inactive(bww1gt_1),384). orientation(bww1gt_2,160,384). coords(bww1gt_2,102,149,384).  visible(bww1gt_2,384). happensAt(walking(bww1gt_2),384).
orientation(bww1gt_1,178,385). coords(bww1gt_1,105,264,385).  visible(bww1gt_1,385). happensAt(inactive(bww1gt_1),385). orientation(bww1gt_2,157,385). coords(bww1gt_2,103,149,385).  visible(bww1gt_2,385). happensAt(walking(bww1gt_2),385).
orientation(bww1gt_1,179,386). coords(bww1gt_1,105,264,386).  visible(bww1gt_1,386). happensAt(inactive(bww1gt_1),386). orientation(bww1gt_2,158,386). coords(bww1gt_2,104,149,386).  visible(bww1gt_2,386). happensAt(walking(bww1gt_2),386).
#show holdsAt/2.