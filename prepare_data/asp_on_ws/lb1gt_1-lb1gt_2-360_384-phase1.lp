
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

	orientation(lb1gt_1,151,360). coords(lb1gt_1,165,170,360).  visible(lb1gt_1,360). happensAt(active(lb1gt_1),360). orientation(lb1gt_2,146,360). coords(lb1gt_2,189,145,360).  visible(lb1gt_2,360). happensAt(active(lb1gt_2),360).
orientation(lb1gt_1,151,361). coords(lb1gt_1,165,170,361).  visible(lb1gt_1,361). happensAt(active(lb1gt_1),361). orientation(lb1gt_2,146,361). coords(lb1gt_2,189,145,361).  visible(lb1gt_2,361). happensAt(active(lb1gt_2),361).
orientation(lb1gt_1,151,362). coords(lb1gt_1,165,170,362).  visible(lb1gt_1,362). happensAt(active(lb1gt_1),362). orientation(lb1gt_2,146,362). coords(lb1gt_2,189,144,362).  visible(lb1gt_2,362). happensAt(active(lb1gt_2),362).
orientation(lb1gt_1,151,363). coords(lb1gt_1,165,170,363).  visible(lb1gt_1,363). happensAt(active(lb1gt_1),363). orientation(lb1gt_2,146,363). coords(lb1gt_2,189,144,363).  visible(lb1gt_2,363). happensAt(active(lb1gt_2),363).
orientation(lb1gt_1,151,364). coords(lb1gt_1,165,170,364).  visible(lb1gt_1,364). happensAt(active(lb1gt_1),364). orientation(lb1gt_2,146,364). coords(lb1gt_2,189,144,364).  visible(lb1gt_2,364). happensAt(active(lb1gt_2),364).
orientation(lb1gt_1,151,365). coords(lb1gt_1,166,170,365).  visible(lb1gt_1,365). happensAt(active(lb1gt_1),365). orientation(lb1gt_2,146,365). coords(lb1gt_2,190,143,365).  visible(lb1gt_2,365). happensAt(active(lb1gt_2),365).
orientation(lb1gt_1,151,366). coords(lb1gt_1,166,170,366).  visible(lb1gt_1,366). happensAt(active(lb1gt_1),366). orientation(lb1gt_2,146,366). coords(lb1gt_2,190,143,366).  visible(lb1gt_2,366). happensAt(active(lb1gt_2),366).
orientation(lb1gt_1,151,367). coords(lb1gt_1,166,170,367).  visible(lb1gt_1,367). happensAt(active(lb1gt_1),367). orientation(lb1gt_2,146,367). coords(lb1gt_2,190,143,367).  visible(lb1gt_2,367). happensAt(active(lb1gt_2),367).
orientation(lb1gt_1,151,368). coords(lb1gt_1,166,170,368).  visible(lb1gt_1,368). happensAt(active(lb1gt_1),368). orientation(lb1gt_2,146,368). coords(lb1gt_2,190,143,368).  visible(lb1gt_2,368). happensAt(active(lb1gt_2),368).
orientation(lb1gt_1,151,369). coords(lb1gt_1,166,170,369).  visible(lb1gt_1,369). happensAt(active(lb1gt_1),369). orientation(lb1gt_2,146,369). coords(lb1gt_2,191,143,369).  visible(lb1gt_2,369). happensAt(active(lb1gt_2),369).
orientation(lb1gt_1,151,370). coords(lb1gt_1,166,170,370).  visible(lb1gt_1,370). happensAt(active(lb1gt_1),370). orientation(lb1gt_2,146,370). coords(lb1gt_2,191,143,370).  visible(lb1gt_2,370). happensAt(active(lb1gt_2),370).
orientation(lb1gt_1,151,371). coords(lb1gt_1,166,170,371).  visible(lb1gt_1,371). happensAt(active(lb1gt_1),371). orientation(lb1gt_2,146,371). coords(lb1gt_2,191,143,371).  visible(lb1gt_2,371). happensAt(active(lb1gt_2),371).
orientation(lb1gt_1,151,372). coords(lb1gt_1,167,170,372).  visible(lb1gt_1,372). happensAt(active(lb1gt_1),372). orientation(lb1gt_2,146,372). coords(lb1gt_2,191,142,372).  visible(lb1gt_2,372). happensAt(active(lb1gt_2),372).
orientation(lb1gt_1,151,373). coords(lb1gt_1,167,170,373).  visible(lb1gt_1,373). happensAt(active(lb1gt_1),373). orientation(lb1gt_2,144,373). coords(lb1gt_2,191,142,373).  visible(lb1gt_2,373). happensAt(active(lb1gt_2),373).
orientation(lb1gt_1,151,374). coords(lb1gt_1,167,170,374).  visible(lb1gt_1,374). happensAt(active(lb1gt_1),374). orientation(lb1gt_2,144,374). coords(lb1gt_2,191,141,374).  visible(lb1gt_2,374). happensAt(active(lb1gt_2),374).
orientation(lb1gt_1,151,375). coords(lb1gt_1,167,170,375).  visible(lb1gt_1,375). happensAt(active(lb1gt_1),375). orientation(lb1gt_2,144,375). coords(lb1gt_2,191,141,375).  visible(lb1gt_2,375). happensAt(active(lb1gt_2),375).
orientation(lb1gt_1,151,376). coords(lb1gt_1,167,170,376).  visible(lb1gt_1,376). happensAt(active(lb1gt_1),376). orientation(lb1gt_2,144,376). coords(lb1gt_2,191,141,376).  visible(lb1gt_2,376). happensAt(active(lb1gt_2),376).
orientation(lb1gt_1,151,377). coords(lb1gt_1,167,170,377).  visible(lb1gt_1,377). happensAt(active(lb1gt_1),377). orientation(lb1gt_2,144,377). coords(lb1gt_2,191,141,377).  visible(lb1gt_2,377). happensAt(active(lb1gt_2),377).
orientation(lb1gt_1,151,378). coords(lb1gt_1,167,169,378).  visible(lb1gt_1,378). happensAt(active(lb1gt_1),378). orientation(lb1gt_2,144,378). coords(lb1gt_2,191,141,378).  visible(lb1gt_2,378). happensAt(active(lb1gt_2),378).
orientation(lb1gt_1,151,379). coords(lb1gt_1,167,169,379).  visible(lb1gt_1,379). happensAt(active(lb1gt_1),379). orientation(lb1gt_2,144,379). coords(lb1gt_2,191,140,379).  visible(lb1gt_2,379). happensAt(active(lb1gt_2),379).
orientation(lb1gt_1,151,380). coords(lb1gt_1,167,169,380).  visible(lb1gt_1,380). happensAt(active(lb1gt_1),380). orientation(lb1gt_2,144,380). coords(lb1gt_2,191,140,380).  visible(lb1gt_2,380). happensAt(active(lb1gt_2),380).
orientation(lb1gt_1,151,381). coords(lb1gt_1,167,169,381).  visible(lb1gt_1,381). happensAt(active(lb1gt_1),381). orientation(lb1gt_2,144,381). coords(lb1gt_2,191,140,381).  visible(lb1gt_2,381). happensAt(active(lb1gt_2),381).
orientation(lb1gt_1,151,382). coords(lb1gt_1,166,169,382).  visible(lb1gt_1,382). happensAt(active(lb1gt_1),382). orientation(lb1gt_2,144,382). coords(lb1gt_2,190,140,382).  visible(lb1gt_2,382). happensAt(active(lb1gt_2),382).
orientation(lb1gt_1,151,383). coords(lb1gt_1,166,169,383).  visible(lb1gt_1,383). happensAt(active(lb1gt_1),383). orientation(lb1gt_2,144,383). coords(lb1gt_2,190,140,383).  visible(lb1gt_2,383). happensAt(active(lb1gt_2),383).
orientation(lb1gt_1,151,384). coords(lb1gt_1,166,168,384).  visible(lb1gt_1,384). happensAt(active(lb1gt_1),384). orientation(lb1gt_2,144,384). coords(lb1gt_2,190,141,384).  visible(lb1gt_2,384). happensAt(active(lb1gt_2),384).
#show holdsAt/2.