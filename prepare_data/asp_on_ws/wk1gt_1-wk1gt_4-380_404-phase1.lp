
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
orientation(wk1gt_1,153,393). coords(wk1gt_1,109,110,393).  visible(wk1gt_1,393). happensAt(walking(wk1gt_1),393). orientation(wk1gt_4,152,393). coords(wk1gt_4,52,173,393).  visible(wk1gt_4,393). happensAt(active(wk1gt_4),393).
orientation(wk1gt_1,147,394). coords(wk1gt_1,108,110,394).  visible(wk1gt_1,394). happensAt(walking(wk1gt_1),394). orientation(wk1gt_4,155,394). coords(wk1gt_4,52,174,394).  visible(wk1gt_4,394). happensAt(active(wk1gt_4),394).
orientation(wk1gt_1,147,395). coords(wk1gt_1,107,109,395).  visible(wk1gt_1,395). happensAt(walking(wk1gt_1),395). orientation(wk1gt_4,158,395). coords(wk1gt_4,52,174,395).  visible(wk1gt_4,395). happensAt(active(wk1gt_4),395).
orientation(wk1gt_1,147,396). coords(wk1gt_1,106,109,396).  visible(wk1gt_1,396). happensAt(walking(wk1gt_1),396). orientation(wk1gt_4,157,396). coords(wk1gt_4,52,174,396).  visible(wk1gt_4,396). happensAt(active(wk1gt_4),396).
orientation(wk1gt_1,154,397). coords(wk1gt_1,105,109,397).  visible(wk1gt_1,397). happensAt(walking(wk1gt_1),397). orientation(wk1gt_4,159,397). coords(wk1gt_4,52,175,397).  visible(wk1gt_4,397). happensAt(active(wk1gt_4),397).
orientation(wk1gt_1,154,398). coords(wk1gt_1,104,109,398).  visible(wk1gt_1,398). happensAt(walking(wk1gt_1),398). orientation(wk1gt_4,157,398). coords(wk1gt_4,52,175,398).  visible(wk1gt_4,398). happensAt(active(wk1gt_4),398).
orientation(wk1gt_1,154,399). coords(wk1gt_1,104,108,399).  visible(wk1gt_1,399). happensAt(walking(wk1gt_1),399). orientation(wk1gt_4,163,399). coords(wk1gt_4,52,176,399).  visible(wk1gt_4,399). happensAt(active(wk1gt_4),399).
orientation(wk1gt_1,154,400). coords(wk1gt_1,103,108,400).  visible(wk1gt_1,400). happensAt(walking(wk1gt_1),400). orientation(wk1gt_4,162,400). coords(wk1gt_4,52,176,400).  visible(wk1gt_4,400). happensAt(active(wk1gt_4),400).
orientation(wk1gt_1,154,401). coords(wk1gt_1,104,108,401).  visible(wk1gt_1,401). happensAt(walking(wk1gt_1),401). orientation(wk1gt_4,168,401). coords(wk1gt_4,52,176,401).  visible(wk1gt_4,401). happensAt(active(wk1gt_4),401).
orientation(wk1gt_1,154,402). coords(wk1gt_1,103,108,402).  visible(wk1gt_1,402). happensAt(walking(wk1gt_1),402). orientation(wk1gt_4,162,402). coords(wk1gt_4,52,176,402).  visible(wk1gt_4,402). happensAt(active(wk1gt_4),402).
orientation(wk1gt_1,148,403). coords(wk1gt_1,103,108,403).  visible(wk1gt_1,403). happensAt(walking(wk1gt_1),403). orientation(wk1gt_4,162,403). coords(wk1gt_4,52,176,403).  visible(wk1gt_4,403). happensAt(active(wk1gt_4),403).
orientation(wk1gt_1,145,404). coords(wk1gt_1,103,107,404).  visible(wk1gt_1,404). happensAt(walking(wk1gt_1),404). orientation(wk1gt_4,160,404). coords(wk1gt_4,52,176,404).  visible(wk1gt_4,404). happensAt(active(wk1gt_4),404).
#show holdsAt/2.