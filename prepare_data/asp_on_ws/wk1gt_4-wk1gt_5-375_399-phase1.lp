
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
orientation(wk1gt_4,155,388). coords(wk1gt_4,52,174,388).  visible(wk1gt_4,388). happensAt(active(wk1gt_4),388). orientation(wk1gt_5,162,388). coords(wk1gt_5,50,163,388).  visible(wk1gt_5,388). happensAt(inactive(wk1gt_5),388).
orientation(wk1gt_4,156,389). coords(wk1gt_4,52,173,389).  visible(wk1gt_4,389). happensAt(active(wk1gt_4),389). orientation(wk1gt_5,162,389). coords(wk1gt_5,50,163,389).  visible(wk1gt_5,389). happensAt(inactive(wk1gt_5),389).
orientation(wk1gt_4,155,390). coords(wk1gt_4,52,173,390).  visible(wk1gt_4,390). happensAt(active(wk1gt_4),390). orientation(wk1gt_5,162,390). coords(wk1gt_5,51,163,390).  visible(wk1gt_5,390). happensAt(inactive(wk1gt_5),390).
orientation(wk1gt_4,162,391). coords(wk1gt_4,52,173,391).  visible(wk1gt_4,391). happensAt(active(wk1gt_4),391). orientation(wk1gt_5,162,391). coords(wk1gt_5,51,163,391).  visible(wk1gt_5,391). happensAt(inactive(wk1gt_5),391).
orientation(wk1gt_4,159,392). coords(wk1gt_4,52,173,392).  visible(wk1gt_4,392). happensAt(active(wk1gt_4),392). orientation(wk1gt_5,162,392). coords(wk1gt_5,50,163,392).  visible(wk1gt_5,392). happensAt(inactive(wk1gt_5),392).
orientation(wk1gt_4,152,393). coords(wk1gt_4,52,173,393).  visible(wk1gt_4,393). happensAt(active(wk1gt_4),393). orientation(wk1gt_5,162,393). coords(wk1gt_5,50,163,393).  visible(wk1gt_5,393). happensAt(inactive(wk1gt_5),393).
orientation(wk1gt_4,155,394). coords(wk1gt_4,52,174,394).  visible(wk1gt_4,394). happensAt(active(wk1gt_4),394). orientation(wk1gt_5,162,394). coords(wk1gt_5,50,163,394).  visible(wk1gt_5,394). happensAt(inactive(wk1gt_5),394).
orientation(wk1gt_4,158,395). coords(wk1gt_4,52,174,395).  visible(wk1gt_4,395). happensAt(active(wk1gt_4),395). orientation(wk1gt_5,162,395). coords(wk1gt_5,50,163,395).  visible(wk1gt_5,395). happensAt(inactive(wk1gt_5),395).
orientation(wk1gt_4,157,396). coords(wk1gt_4,52,174,396).  visible(wk1gt_4,396). happensAt(active(wk1gt_4),396). orientation(wk1gt_5,162,396). coords(wk1gt_5,50,163,396).  visible(wk1gt_5,396). happensAt(inactive(wk1gt_5),396).
orientation(wk1gt_4,159,397). coords(wk1gt_4,52,175,397).  visible(wk1gt_4,397). happensAt(active(wk1gt_4),397). orientation(wk1gt_5,162,397). coords(wk1gt_5,50,163,397).  visible(wk1gt_5,397). happensAt(inactive(wk1gt_5),397).
orientation(wk1gt_4,157,398). coords(wk1gt_4,52,175,398).  visible(wk1gt_4,398). happensAt(active(wk1gt_4),398). orientation(wk1gt_5,162,398). coords(wk1gt_5,50,163,398).  visible(wk1gt_5,398). happensAt(inactive(wk1gt_5),398).
orientation(wk1gt_4,163,399). coords(wk1gt_4,52,176,399).  visible(wk1gt_4,399). happensAt(active(wk1gt_4),399). orientation(wk1gt_5,162,399). coords(wk1gt_5,50,163,399).  visible(wk1gt_5,399). happensAt(inactive(wk1gt_5),399).
#show holdsAt/2.