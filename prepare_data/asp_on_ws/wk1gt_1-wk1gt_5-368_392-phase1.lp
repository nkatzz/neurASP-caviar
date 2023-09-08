
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

	orientation(wk1gt_1,153,368). coords(wk1gt_1,126,121,368).  visible(wk1gt_1,368). happensAt(walking(wk1gt_1),368). orientation(wk1gt_5,162,368). coords(wk1gt_5,51,163,368).  visible(wk1gt_5,368). happensAt(inactive(wk1gt_5),368).
orientation(wk1gt_1,153,369). coords(wk1gt_1,124,120,369).  visible(wk1gt_1,369). happensAt(walking(wk1gt_1),369). orientation(wk1gt_5,162,369). coords(wk1gt_5,50,163,369).  visible(wk1gt_5,369). happensAt(inactive(wk1gt_5),369).
orientation(wk1gt_1,153,370). coords(wk1gt_1,124,120,370).  visible(wk1gt_1,370). happensAt(walking(wk1gt_1),370). orientation(wk1gt_5,162,370). coords(wk1gt_5,50,163,370).  visible(wk1gt_5,370). happensAt(inactive(wk1gt_5),370).
orientation(wk1gt_1,153,371). coords(wk1gt_1,123,119,371).  visible(wk1gt_1,371). happensAt(walking(wk1gt_1),371). orientation(wk1gt_5,162,371). coords(wk1gt_5,50,163,371).  visible(wk1gt_5,371). happensAt(inactive(wk1gt_5),371).
orientation(wk1gt_1,153,372). coords(wk1gt_1,120,119,372).  visible(wk1gt_1,372). happensAt(walking(wk1gt_1),372). orientation(wk1gt_5,162,372). coords(wk1gt_5,50,163,372).  visible(wk1gt_5,372). happensAt(inactive(wk1gt_5),372).
orientation(wk1gt_1,149,373). coords(wk1gt_1,119,119,373).  visible(wk1gt_1,373). happensAt(walking(wk1gt_1),373). orientation(wk1gt_5,162,373). coords(wk1gt_5,50,163,373).  visible(wk1gt_5,373). happensAt(inactive(wk1gt_5),373).
orientation(wk1gt_1,149,374). coords(wk1gt_1,119,119,374).  visible(wk1gt_1,374). happensAt(walking(wk1gt_1),374). orientation(wk1gt_5,162,374). coords(wk1gt_5,50,163,374).  visible(wk1gt_5,374). happensAt(inactive(wk1gt_5),374).
orientation(wk1gt_1,149,375). coords(wk1gt_1,118,118,375).  visible(wk1gt_1,375). happensAt(walking(wk1gt_1),375). orientation(wk1gt_5,162,375). coords(wk1gt_5,50,163,375).  visible(wk1gt_5,375). happensAt(inactive(wk1gt_5),375).
orientation(wk1gt_1,149,376). coords(wk1gt_1,118,118,376).  visible(wk1gt_1,376). happensAt(walking(wk1gt_1),376). orientation(wk1gt_5,162,376). coords(wk1gt_5,50,163,376).  visible(wk1gt_5,376). happensAt(inactive(wk1gt_5),376).
orientation(wk1gt_1,149,377). coords(wk1gt_1,118,118,377).  visible(wk1gt_1,377). happensAt(walking(wk1gt_1),377). orientation(wk1gt_5,162,377). coords(wk1gt_5,51,163,377).  visible(wk1gt_5,377). happensAt(inactive(wk1gt_5),377).
orientation(wk1gt_1,149,378). coords(wk1gt_1,118,118,378).  visible(wk1gt_1,378). happensAt(walking(wk1gt_1),378). orientation(wk1gt_5,162,378). coords(wk1gt_5,50,163,378).  visible(wk1gt_5,378). happensAt(inactive(wk1gt_5),378).
orientation(wk1gt_1,149,379). coords(wk1gt_1,118,118,379).  visible(wk1gt_1,379). happensAt(walking(wk1gt_1),379). orientation(wk1gt_5,162,379). coords(wk1gt_5,50,163,379).  visible(wk1gt_5,379). happensAt(inactive(wk1gt_5),379).
orientation(wk1gt_1,147,380). coords(wk1gt_1,117,117,380).  visible(wk1gt_1,380). happensAt(walking(wk1gt_1),380). orientation(wk1gt_5,162,380). coords(wk1gt_5,50,163,380).  visible(wk1gt_5,380). happensAt(inactive(wk1gt_5),380).
orientation(wk1gt_1,147,381). coords(wk1gt_1,116,117,381).  visible(wk1gt_1,381). happensAt(walking(wk1gt_1),381). orientation(wk1gt_5,162,381). coords(wk1gt_5,50,163,381).  visible(wk1gt_5,381). happensAt(inactive(wk1gt_5),381).
orientation(wk1gt_1,147,382). coords(wk1gt_1,115,116,382).  visible(wk1gt_1,382). happensAt(walking(wk1gt_1),382). orientation(wk1gt_5,162,382). coords(wk1gt_5,50,163,382).  visible(wk1gt_5,382). happensAt(inactive(wk1gt_5),382).
orientation(wk1gt_1,147,383). coords(wk1gt_1,113,115,383).  visible(wk1gt_1,383). happensAt(walking(wk1gt_1),383). orientation(wk1gt_5,162,383). coords(wk1gt_5,50,163,383).  visible(wk1gt_5,383). happensAt(inactive(wk1gt_5),383).
orientation(wk1gt_1,147,384). coords(wk1gt_1,112,115,384).  visible(wk1gt_1,384). happensAt(walking(wk1gt_1),384). orientation(wk1gt_5,162,384). coords(wk1gt_5,50,163,384).  visible(wk1gt_5,384). happensAt(inactive(wk1gt_5),384).
orientation(wk1gt_1,147,385). coords(wk1gt_1,112,115,385).  visible(wk1gt_1,385). happensAt(walking(wk1gt_1),385). orientation(wk1gt_5,162,385). coords(wk1gt_5,50,163,385).  visible(wk1gt_5,385). happensAt(inactive(wk1gt_5),385).
orientation(wk1gt_1,147,386). coords(wk1gt_1,112,114,386).  visible(wk1gt_1,386). happensAt(walking(wk1gt_1),386). orientation(wk1gt_5,162,386). coords(wk1gt_5,50,163,386).  visible(wk1gt_5,386). happensAt(inactive(wk1gt_5),386).
orientation(wk1gt_1,147,387). coords(wk1gt_1,111,112,387).  visible(wk1gt_1,387). happensAt(walking(wk1gt_1),387). orientation(wk1gt_5,162,387). coords(wk1gt_5,50,163,387).  visible(wk1gt_5,387). happensAt(inactive(wk1gt_5),387).
orientation(wk1gt_1,147,388). coords(wk1gt_1,111,112,388).  visible(wk1gt_1,388). happensAt(walking(wk1gt_1),388). orientation(wk1gt_5,162,388). coords(wk1gt_5,50,163,388).  visible(wk1gt_5,388). happensAt(inactive(wk1gt_5),388).
orientation(wk1gt_1,147,389). coords(wk1gt_1,111,112,389).  visible(wk1gt_1,389). happensAt(walking(wk1gt_1),389). orientation(wk1gt_5,162,389). coords(wk1gt_5,50,163,389).  visible(wk1gt_5,389). happensAt(inactive(wk1gt_5),389).
orientation(wk1gt_1,147,390). coords(wk1gt_1,111,112,390).  visible(wk1gt_1,390). happensAt(walking(wk1gt_1),390). orientation(wk1gt_5,162,390). coords(wk1gt_5,51,163,390).  visible(wk1gt_5,390). happensAt(inactive(wk1gt_5),390).
orientation(wk1gt_1,147,391). coords(wk1gt_1,110,112,391).  visible(wk1gt_1,391). happensAt(walking(wk1gt_1),391). orientation(wk1gt_5,162,391). coords(wk1gt_5,51,163,391).  visible(wk1gt_5,391). happensAt(inactive(wk1gt_5),391).
orientation(wk1gt_1,147,392). coords(wk1gt_1,110,111,392).  visible(wk1gt_1,392). happensAt(walking(wk1gt_1),392). orientation(wk1gt_5,162,392). coords(wk1gt_5,50,163,392).  visible(wk1gt_5,392). happensAt(inactive(wk1gt_5),392).
#show holdsAt/2.