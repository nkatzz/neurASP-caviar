
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

	orientation(wk1gt_4,164,363). coords(wk1gt_4,52,176,363).  visible(wk1gt_4,363). happensAt(inactive(wk1gt_4),363). orientation(wk1gt_5,162,363). coords(wk1gt_5,50,163,363).  visible(wk1gt_5,363). happensAt(inactive(wk1gt_5),363).
orientation(wk1gt_4,166,364). coords(wk1gt_4,52,176,364).  visible(wk1gt_4,364). happensAt(inactive(wk1gt_4),364). orientation(wk1gt_5,162,364). coords(wk1gt_5,51,163,364).  visible(wk1gt_5,364). happensAt(inactive(wk1gt_5),364).
orientation(wk1gt_4,162,365). coords(wk1gt_4,52,176,365).  visible(wk1gt_4,365). happensAt(inactive(wk1gt_4),365). orientation(wk1gt_5,162,365). coords(wk1gt_5,50,163,365).  visible(wk1gt_5,365). happensAt(inactive(wk1gt_5),365).
orientation(wk1gt_4,164,366). coords(wk1gt_4,52,176,366).  visible(wk1gt_4,366). happensAt(inactive(wk1gt_4),366). orientation(wk1gt_5,162,366). coords(wk1gt_5,50,163,366).  visible(wk1gt_5,366). happensAt(inactive(wk1gt_5),366).
orientation(wk1gt_4,162,367). coords(wk1gt_4,52,176,367).  visible(wk1gt_4,367). happensAt(inactive(wk1gt_4),367). orientation(wk1gt_5,162,367). coords(wk1gt_5,50,163,367).  visible(wk1gt_5,367). happensAt(inactive(wk1gt_5),367).
orientation(wk1gt_4,161,368). coords(wk1gt_4,52,176,368).  visible(wk1gt_4,368). happensAt(inactive(wk1gt_4),368). orientation(wk1gt_5,162,368). coords(wk1gt_5,51,163,368).  visible(wk1gt_5,368). happensAt(inactive(wk1gt_5),368).
orientation(wk1gt_4,162,369). coords(wk1gt_4,52,176,369).  visible(wk1gt_4,369). happensAt(inactive(wk1gt_4),369). orientation(wk1gt_5,162,369). coords(wk1gt_5,50,163,369).  visible(wk1gt_5,369). happensAt(inactive(wk1gt_5),369).
orientation(wk1gt_4,163,370). coords(wk1gt_4,52,176,370).  visible(wk1gt_4,370). happensAt(inactive(wk1gt_4),370). orientation(wk1gt_5,162,370). coords(wk1gt_5,50,163,370).  visible(wk1gt_5,370). happensAt(inactive(wk1gt_5),370).
orientation(wk1gt_4,164,371). coords(wk1gt_4,52,176,371).  visible(wk1gt_4,371). happensAt(inactive(wk1gt_4),371). orientation(wk1gt_5,162,371). coords(wk1gt_5,50,163,371).  visible(wk1gt_5,371). happensAt(inactive(wk1gt_5),371).
orientation(wk1gt_4,163,372). coords(wk1gt_4,52,176,372).  visible(wk1gt_4,372). happensAt(inactive(wk1gt_4),372). orientation(wk1gt_5,162,372). coords(wk1gt_5,50,163,372).  visible(wk1gt_5,372). happensAt(inactive(wk1gt_5),372).
orientation(wk1gt_4,164,373). coords(wk1gt_4,52,176,373).  visible(wk1gt_4,373). happensAt(inactive(wk1gt_4),373). orientation(wk1gt_5,162,373). coords(wk1gt_5,50,163,373).  visible(wk1gt_5,373). happensAt(inactive(wk1gt_5),373).
orientation(wk1gt_4,166,374). coords(wk1gt_4,52,176,374).  visible(wk1gt_4,374). happensAt(inactive(wk1gt_4),374). orientation(wk1gt_5,162,374). coords(wk1gt_5,50,163,374).  visible(wk1gt_5,374). happensAt(inactive(wk1gt_5),374).
orientation(wk1gt_4,170,375). coords(wk1gt_4,52,176,375).  visible(wk1gt_4,375). happensAt(inactive(wk1gt_4),375). orientation(wk1gt_5,162,375). coords(wk1gt_5,50,163,375).  visible(wk1gt_5,375). happensAt(inactive(wk1gt_5),375).
orientation(wk1gt_4,163,376). coords(wk1gt_4,52,176,376).  visible(wk1gt_4,376). happensAt(inactive(wk1gt_4),376). orientation(wk1gt_5,162,376). coords(wk1gt_5,50,163,376).  visible(wk1gt_5,376). happensAt(inactive(wk1gt_5),376).
orientation(wk1gt_4,163,377). coords(wk1gt_4,52,176,377).  visible(wk1gt_4,377). happensAt(inactive(wk1gt_4),377). orientation(wk1gt_5,162,377). coords(wk1gt_5,51,163,377).  visible(wk1gt_5,377). happensAt(inactive(wk1gt_5),377).
orientation(wk1gt_4,161,378). coords(wk1gt_4,52,176,378).  visible(wk1gt_4,378). happensAt(inactive(wk1gt_4),378). orientation(wk1gt_5,162,378). coords(wk1gt_5,50,163,378).  visible(wk1gt_5,378). happensAt(inactive(wk1gt_5),378).
orientation(wk1gt_4,154,379). coords(wk1gt_4,52,176,379).  visible(wk1gt_4,379). happensAt(inactive(wk1gt_4),379). orientation(wk1gt_5,162,379). coords(wk1gt_5,50,163,379).  visible(wk1gt_5,379). happensAt(inactive(wk1gt_5),379).
orientation(wk1gt_4,157,380). coords(wk1gt_4,52,176,380).  visible(wk1gt_4,380). happensAt(inactive(wk1gt_4),380). orientation(wk1gt_5,162,380). coords(wk1gt_5,50,163,380).  visible(wk1gt_5,380). happensAt(inactive(wk1gt_5),380).
orientation(wk1gt_4,159,381). coords(wk1gt_4,52,176,381).  visible(wk1gt_4,381). happensAt(inactive(wk1gt_4),381). orientation(wk1gt_5,162,381). coords(wk1gt_5,50,163,381).  visible(wk1gt_5,381). happensAt(inactive(wk1gt_5),381).
orientation(wk1gt_4,155,382). coords(wk1gt_4,52,176,382).  visible(wk1gt_4,382). happensAt(inactive(wk1gt_4),382). orientation(wk1gt_5,162,382). coords(wk1gt_5,50,163,382).  visible(wk1gt_5,382). happensAt(inactive(wk1gt_5),382).
orientation(wk1gt_4,159,383). coords(wk1gt_4,52,176,383).  visible(wk1gt_4,383). happensAt(inactive(wk1gt_4),383). orientation(wk1gt_5,162,383). coords(wk1gt_5,50,163,383).  visible(wk1gt_5,383). happensAt(inactive(wk1gt_5),383).
orientation(wk1gt_4,152,384). coords(wk1gt_4,52,175,384).  visible(wk1gt_4,384). happensAt(inactive(wk1gt_4),384). orientation(wk1gt_5,162,384). coords(wk1gt_5,50,163,384).  visible(wk1gt_5,384). happensAt(inactive(wk1gt_5),384).
orientation(wk1gt_4,154,385). coords(wk1gt_4,52,175,385).  visible(wk1gt_4,385). happensAt(active(wk1gt_4),385). orientation(wk1gt_5,162,385). coords(wk1gt_5,50,163,385).  visible(wk1gt_5,385). happensAt(inactive(wk1gt_5),385).
orientation(wk1gt_4,155,386). coords(wk1gt_4,52,174,386).  visible(wk1gt_4,386). happensAt(active(wk1gt_4),386). orientation(wk1gt_5,162,386). coords(wk1gt_5,50,163,386).  visible(wk1gt_5,386). happensAt(inactive(wk1gt_5),386).
orientation(wk1gt_4,156,387). coords(wk1gt_4,52,174,387).  visible(wk1gt_4,387). happensAt(active(wk1gt_4),387). orientation(wk1gt_5,162,387). coords(wk1gt_5,50,163,387).  visible(wk1gt_5,387). happensAt(inactive(wk1gt_5),387).
#show holdsAt/2.