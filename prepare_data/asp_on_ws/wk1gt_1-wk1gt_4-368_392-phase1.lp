
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

	orientation(wk1gt_1,153,368). coords(wk1gt_1,126,121,368).  visible(wk1gt_1,368). happensAt(walking(wk1gt_1),368). orientation(wk1gt_4,161,368). coords(wk1gt_4,52,176,368).  visible(wk1gt_4,368). happensAt(inactive(wk1gt_4),368).
orientation(wk1gt_1,153,369). coords(wk1gt_1,124,120,369).  visible(wk1gt_1,369). happensAt(walking(wk1gt_1),369). orientation(wk1gt_4,162,369). coords(wk1gt_4,52,176,369).  visible(wk1gt_4,369). happensAt(inactive(wk1gt_4),369).
orientation(wk1gt_1,153,370). coords(wk1gt_1,124,120,370).  visible(wk1gt_1,370). happensAt(walking(wk1gt_1),370). orientation(wk1gt_4,163,370). coords(wk1gt_4,52,176,370).  visible(wk1gt_4,370). happensAt(inactive(wk1gt_4),370).
orientation(wk1gt_1,153,371). coords(wk1gt_1,123,119,371).  visible(wk1gt_1,371). happensAt(walking(wk1gt_1),371). orientation(wk1gt_4,164,371). coords(wk1gt_4,52,176,371).  visible(wk1gt_4,371). happensAt(inactive(wk1gt_4),371).
orientation(wk1gt_1,153,372). coords(wk1gt_1,120,119,372).  visible(wk1gt_1,372). happensAt(walking(wk1gt_1),372). orientation(wk1gt_4,163,372). coords(wk1gt_4,52,176,372).  visible(wk1gt_4,372). happensAt(inactive(wk1gt_4),372).
orientation(wk1gt_1,149,373). coords(wk1gt_1,119,119,373).  visible(wk1gt_1,373). happensAt(walking(wk1gt_1),373). orientation(wk1gt_4,164,373). coords(wk1gt_4,52,176,373).  visible(wk1gt_4,373). happensAt(inactive(wk1gt_4),373).
orientation(wk1gt_1,149,374). coords(wk1gt_1,119,119,374).  visible(wk1gt_1,374). happensAt(walking(wk1gt_1),374). orientation(wk1gt_4,166,374). coords(wk1gt_4,52,176,374).  visible(wk1gt_4,374). happensAt(inactive(wk1gt_4),374).
orientation(wk1gt_1,149,375). coords(wk1gt_1,118,118,375).  visible(wk1gt_1,375). happensAt(walking(wk1gt_1),375). orientation(wk1gt_4,170,375). coords(wk1gt_4,52,176,375).  visible(wk1gt_4,375). happensAt(inactive(wk1gt_4),375).
orientation(wk1gt_1,149,376). coords(wk1gt_1,118,118,376).  visible(wk1gt_1,376). happensAt(walking(wk1gt_1),376). orientation(wk1gt_4,163,376). coords(wk1gt_4,52,176,376).  visible(wk1gt_4,376). happensAt(inactive(wk1gt_4),376).
orientation(wk1gt_1,149,377). coords(wk1gt_1,118,118,377).  visible(wk1gt_1,377). happensAt(walking(wk1gt_1),377). orientation(wk1gt_4,163,377). coords(wk1gt_4,52,176,377).  visible(wk1gt_4,377). happensAt(inactive(wk1gt_4),377).
orientation(wk1gt_1,149,378). coords(wk1gt_1,118,118,378).  visible(wk1gt_1,378). happensAt(walking(wk1gt_1),378). orientation(wk1gt_4,161,378). coords(wk1gt_4,52,176,378).  visible(wk1gt_4,378). happensAt(inactive(wk1gt_4),378).
orientation(wk1gt_1,149,379). coords(wk1gt_1,118,118,379).  visible(wk1gt_1,379). happensAt(walking(wk1gt_1),379). orientation(wk1gt_4,154,379). coords(wk1gt_4,52,176,379).  visible(wk1gt_4,379). happensAt(inactive(wk1gt_4),379).
orientation(wk1gt_1,147,380). coords(wk1gt_1,117,117,380).  visible(wk1gt_1,380). happensAt(walking(wk1gt_1),380). orientation(wk1gt_4,157,380). coords(wk1gt_4,52,176,380).  visible(wk1gt_4,380). happensAt(inactive(wk1gt_4),380).
orientation(wk1gt_1,147,381). coords(wk1gt_1,116,117,381).  visible(wk1gt_1,381). happensAt(walking(wk1gt_1),381). orientation(wk1gt_4,159,381). coords(wk1gt_4,52,176,381).  visible(wk1gt_4,381). happensAt(inactive(wk1gt_4),381).
orientation(wk1gt_1,147,382). coords(wk1gt_1,115,116,382).  visible(wk1gt_1,382). happensAt(walking(wk1gt_1),382). orientation(wk1gt_4,155,382). coords(wk1gt_4,52,176,382).  visible(wk1gt_4,382). happensAt(inactive(wk1gt_4),382).
orientation(wk1gt_1,147,383). coords(wk1gt_1,113,115,383).  visible(wk1gt_1,383). happensAt(walking(wk1gt_1),383). orientation(wk1gt_4,159,383). coords(wk1gt_4,52,176,383).  visible(wk1gt_4,383). happensAt(inactive(wk1gt_4),383).
orientation(wk1gt_1,147,384). coords(wk1gt_1,112,115,384).  visible(wk1gt_1,384). happensAt(walking(wk1gt_1),384). orientation(wk1gt_4,152,384). coords(wk1gt_4,52,175,384).  visible(wk1gt_4,384). happensAt(inactive(wk1gt_4),384).
orientation(wk1gt_1,147,385). coords(wk1gt_1,112,115,385).  visible(wk1gt_1,385). happensAt(walking(wk1gt_1),385). orientation(wk1gt_4,154,385). coords(wk1gt_4,52,175,385).  visible(wk1gt_4,385). happensAt(active(wk1gt_4),385).
orientation(wk1gt_1,147,386). coords(wk1gt_1,112,114,386).  visible(wk1gt_1,386). happensAt(walking(wk1gt_1),386). orientation(wk1gt_4,155,386). coords(wk1gt_4,52,174,386).  visible(wk1gt_4,386). happensAt(active(wk1gt_4),386).
orientation(wk1gt_1,147,387). coords(wk1gt_1,111,112,387).  visible(wk1gt_1,387). happensAt(walking(wk1gt_1),387). orientation(wk1gt_4,156,387). coords(wk1gt_4,52,174,387).  visible(wk1gt_4,387). happensAt(active(wk1gt_4),387).
orientation(wk1gt_1,147,388). coords(wk1gt_1,111,112,388).  visible(wk1gt_1,388). happensAt(walking(wk1gt_1),388). orientation(wk1gt_4,155,388). coords(wk1gt_4,52,174,388).  visible(wk1gt_4,388). happensAt(active(wk1gt_4),388).
orientation(wk1gt_1,147,389). coords(wk1gt_1,111,112,389).  visible(wk1gt_1,389). happensAt(walking(wk1gt_1),389). orientation(wk1gt_4,156,389). coords(wk1gt_4,52,173,389).  visible(wk1gt_4,389). happensAt(active(wk1gt_4),389).
orientation(wk1gt_1,147,390). coords(wk1gt_1,111,112,390).  visible(wk1gt_1,390). happensAt(walking(wk1gt_1),390). orientation(wk1gt_4,155,390). coords(wk1gt_4,52,173,390).  visible(wk1gt_4,390). happensAt(active(wk1gt_4),390).
orientation(wk1gt_1,147,391). coords(wk1gt_1,110,112,391).  visible(wk1gt_1,391). happensAt(walking(wk1gt_1),391). orientation(wk1gt_4,162,391). coords(wk1gt_4,52,173,391).  visible(wk1gt_4,391). happensAt(active(wk1gt_4),391).
orientation(wk1gt_1,147,392). coords(wk1gt_1,110,111,392).  visible(wk1gt_1,392). happensAt(walking(wk1gt_1),392). orientation(wk1gt_4,159,392). coords(wk1gt_4,52,173,392).  visible(wk1gt_4,392). happensAt(active(wk1gt_4),392).
#show holdsAt/2.