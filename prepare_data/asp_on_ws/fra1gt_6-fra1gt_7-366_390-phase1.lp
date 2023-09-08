
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

	orientation(fra1gt_6,126,366). coords(fra1gt_6,155,88,366).  visible(fra1gt_6,366). happensAt(walking(fra1gt_6),366). orientation(fra1gt_7,160,366). coords(fra1gt_7,173,70,366).  visible(fra1gt_7,366). happensAt(walking(fra1gt_7),366).
orientation(fra1gt_6,126,367). coords(fra1gt_6,155,88,367).  visible(fra1gt_6,367). happensAt(walking(fra1gt_6),367). orientation(fra1gt_7,160,367). coords(fra1gt_7,173,70,367).  visible(fra1gt_7,367). happensAt(walking(fra1gt_7),367).
orientation(fra1gt_6,126,368). coords(fra1gt_6,154,88,368).  visible(fra1gt_6,368). happensAt(walking(fra1gt_6),368). orientation(fra1gt_7,160,368). coords(fra1gt_7,173,70,368).  visible(fra1gt_7,368). happensAt(walking(fra1gt_7),368).
orientation(fra1gt_6,126,369). coords(fra1gt_6,153,88,369).  visible(fra1gt_6,369). happensAt(walking(fra1gt_6),369). orientation(fra1gt_7,162,369). coords(fra1gt_7,173,71,369).  visible(fra1gt_7,369). happensAt(walking(fra1gt_7),369).
orientation(fra1gt_6,126,370). coords(fra1gt_6,153,88,370).  visible(fra1gt_6,370). happensAt(walking(fra1gt_6),370). orientation(fra1gt_7,162,370). coords(fra1gt_7,173,71,370).  visible(fra1gt_7,370). happensAt(walking(fra1gt_7),370).
orientation(fra1gt_6,126,371). coords(fra1gt_6,152,87,371).  visible(fra1gt_6,371). happensAt(walking(fra1gt_6),371). orientation(fra1gt_7,162,371). coords(fra1gt_7,172,69,371).  visible(fra1gt_7,371). happensAt(walking(fra1gt_7),371).
orientation(fra1gt_6,126,372). coords(fra1gt_6,152,88,372).  visible(fra1gt_6,372). happensAt(walking(fra1gt_6),372). orientation(fra1gt_7,162,372). coords(fra1gt_7,169,69,372).  visible(fra1gt_7,372). happensAt(walking(fra1gt_7),372).
orientation(fra1gt_6,136,373). coords(fra1gt_6,155,88,373).  visible(fra1gt_6,373). happensAt(walking(fra1gt_6),373). orientation(fra1gt_7,162,373). coords(fra1gt_7,167,69,373).  visible(fra1gt_7,373). happensAt(running(fra1gt_7),373).
orientation(fra1gt_6,136,374). coords(fra1gt_6,155,89,374).  visible(fra1gt_6,374). happensAt(walking(fra1gt_6),374). orientation(fra1gt_7,162,374). coords(fra1gt_7,164,69,374).  visible(fra1gt_7,374). happensAt(running(fra1gt_7),374).
orientation(fra1gt_6,136,375). coords(fra1gt_6,155,90,375).  visible(fra1gt_6,375). happensAt(walking(fra1gt_6),375). orientation(fra1gt_7,162,375). coords(fra1gt_7,164,70,375).  visible(fra1gt_7,375). happensAt(running(fra1gt_7),375).
orientation(fra1gt_6,136,376). coords(fra1gt_6,155,90,376).  visible(fra1gt_6,376). happensAt(walking(fra1gt_6),376). orientation(fra1gt_7,162,376). coords(fra1gt_7,162,71,376).  visible(fra1gt_7,376). happensAt(running(fra1gt_7),376).
orientation(fra1gt_6,136,377). coords(fra1gt_6,155,92,377).  visible(fra1gt_6,377). happensAt(walking(fra1gt_6),377). orientation(fra1gt_7,162,377). coords(fra1gt_7,159,71,377).  visible(fra1gt_7,377). happensAt(running(fra1gt_7),377).
orientation(fra1gt_6,136,378). coords(fra1gt_6,156,93,378).  visible(fra1gt_6,378). happensAt(walking(fra1gt_6),378). orientation(fra1gt_7,162,378). coords(fra1gt_7,157,70,378).  visible(fra1gt_7,378). happensAt(running(fra1gt_7),378).
orientation(fra1gt_6,136,379). coords(fra1gt_6,155,93,379).  visible(fra1gt_6,379). happensAt(walking(fra1gt_6),379). orientation(fra1gt_7,162,379). coords(fra1gt_7,153,71,379).  visible(fra1gt_7,379). happensAt(running(fra1gt_7),379).
orientation(fra1gt_6,146,380). coords(fra1gt_6,154,94,380).  visible(fra1gt_6,380). happensAt(walking(fra1gt_6),380). orientation(fra1gt_7,162,380). coords(fra1gt_7,152,71,380).  visible(fra1gt_7,380). happensAt(running(fra1gt_7),380).
orientation(fra1gt_6,146,381). coords(fra1gt_6,153,94,381).  visible(fra1gt_6,381). happensAt(walking(fra1gt_6),381). orientation(fra1gt_7,162,381). coords(fra1gt_7,150,69,381).  visible(fra1gt_7,381). happensAt(running(fra1gt_7),381).
orientation(fra1gt_6,146,382). coords(fra1gt_6,153,94,382).  visible(fra1gt_6,382). happensAt(walking(fra1gt_6),382). orientation(fra1gt_7,162,382). coords(fra1gt_7,149,69,382).  visible(fra1gt_7,382). happensAt(running(fra1gt_7),382).
orientation(fra1gt_6,146,383). coords(fra1gt_6,152,95,383).  visible(fra1gt_6,383). happensAt(walking(fra1gt_6),383). orientation(fra1gt_7,162,383). coords(fra1gt_7,146,69,383).  visible(fra1gt_7,383). happensAt(running(fra1gt_7),383).
orientation(fra1gt_6,151,384). coords(fra1gt_6,151,95,384).  visible(fra1gt_6,384). happensAt(walking(fra1gt_6),384). orientation(fra1gt_7,162,384). coords(fra1gt_7,144,68,384).  visible(fra1gt_7,384). happensAt(running(fra1gt_7),384).
orientation(fra1gt_6,151,385). coords(fra1gt_6,148,96,385).  visible(fra1gt_6,385). happensAt(running(fra1gt_6),385). orientation(fra1gt_7,162,385). coords(fra1gt_7,141,68,385).  visible(fra1gt_7,385). happensAt(running(fra1gt_7),385).
orientation(fra1gt_6,151,386). coords(fra1gt_6,146,95,386).  visible(fra1gt_6,386). happensAt(running(fra1gt_6),386). orientation(fra1gt_7,162,386). coords(fra1gt_7,137,69,386).  visible(fra1gt_7,386). happensAt(running(fra1gt_7),386).
orientation(fra1gt_6,160,387). coords(fra1gt_6,144,97,387).  visible(fra1gt_6,387). happensAt(running(fra1gt_6),387). orientation(fra1gt_7,162,387). coords(fra1gt_7,135,70,387).  visible(fra1gt_7,387). happensAt(running(fra1gt_7),387).
orientation(fra1gt_6,160,388). coords(fra1gt_6,143,97,388).  visible(fra1gt_6,388). happensAt(running(fra1gt_6),388). orientation(fra1gt_7,162,388). coords(fra1gt_7,133,72,388).  visible(fra1gt_7,388). happensAt(running(fra1gt_7),388).
orientation(fra1gt_6,160,389). coords(fra1gt_6,141,99,389).  visible(fra1gt_6,389). happensAt(running(fra1gt_6),389). orientation(fra1gt_7,162,389). coords(fra1gt_7,131,73,389).  visible(fra1gt_7,389). happensAt(running(fra1gt_7),389).
orientation(fra1gt_6,160,390). coords(fra1gt_6,141,99,390).  visible(fra1gt_6,390). happensAt(running(fra1gt_6),390). orientation(fra1gt_7,162,390). coords(fra1gt_7,129,74,390).  visible(fra1gt_7,390). happensAt(running(fra1gt_7),390).
#show holdsAt/2.