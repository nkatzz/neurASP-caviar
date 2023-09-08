
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

	orientation(mwt1gt_1,160,377). coords(mwt1gt_1,105,135,377).  visible(mwt1gt_1,377). happensAt(walking(mwt1gt_1),377). orientation(mwt1gt_2,140,377). coords(mwt1gt_2,105,114,377).  visible(mwt1gt_2,377). happensAt(walking(mwt1gt_2),377).
orientation(mwt1gt_1,160,378). coords(mwt1gt_1,103,132,378).  visible(mwt1gt_1,378). happensAt(walking(mwt1gt_1),378). orientation(mwt1gt_2,140,378). coords(mwt1gt_2,105,114,378).  visible(mwt1gt_2,378). happensAt(walking(mwt1gt_2),378).
orientation(mwt1gt_1,160,379). coords(mwt1gt_1,103,132,379).  visible(mwt1gt_1,379). happensAt(walking(mwt1gt_1),379). orientation(mwt1gt_2,140,379). coords(mwt1gt_2,105,114,379).  visible(mwt1gt_2,379). happensAt(walking(mwt1gt_2),379).
orientation(mwt1gt_1,160,380). coords(mwt1gt_1,100,130,380).  visible(mwt1gt_1,380). happensAt(walking(mwt1gt_1),380). orientation(mwt1gt_2,140,380). coords(mwt1gt_2,103,113,380).  visible(mwt1gt_2,380). happensAt(walking(mwt1gt_2),380).
orientation(mwt1gt_1,160,381). coords(mwt1gt_1,100,129,381).  visible(mwt1gt_1,381). happensAt(walking(mwt1gt_1),381). orientation(mwt1gt_2,140,381). coords(mwt1gt_2,103,113,381).  visible(mwt1gt_2,381). happensAt(walking(mwt1gt_2),381).
orientation(mwt1gt_1,160,382). coords(mwt1gt_1,100,129,382).  visible(mwt1gt_1,382). happensAt(walking(mwt1gt_1),382). orientation(mwt1gt_2,140,382). coords(mwt1gt_2,101,111,382).  visible(mwt1gt_2,382). happensAt(walking(mwt1gt_2),382).
orientation(mwt1gt_1,160,383). coords(mwt1gt_1,100,129,383).  visible(mwt1gt_1,383). happensAt(walking(mwt1gt_1),383). orientation(mwt1gt_2,140,383). coords(mwt1gt_2,101,111,383).  visible(mwt1gt_2,383). happensAt(walking(mwt1gt_2),383).
orientation(mwt1gt_1,160,384). coords(mwt1gt_1,100,129,384).  visible(mwt1gt_1,384). happensAt(walking(mwt1gt_1),384). orientation(mwt1gt_2,140,384). coords(mwt1gt_2,101,111,384).  visible(mwt1gt_2,384). happensAt(walking(mwt1gt_2),384).
orientation(mwt1gt_1,160,385). coords(mwt1gt_1,97,126,385).  visible(mwt1gt_1,385). happensAt(walking(mwt1gt_1),385). orientation(mwt1gt_2,140,385). coords(mwt1gt_2,98,109,385).  visible(mwt1gt_2,385). happensAt(walking(mwt1gt_2),385).
orientation(mwt1gt_1,160,386). coords(mwt1gt_1,97,126,386).  visible(mwt1gt_1,386). happensAt(walking(mwt1gt_1),386). orientation(mwt1gt_2,140,386). coords(mwt1gt_2,98,109,386).  visible(mwt1gt_2,386). happensAt(walking(mwt1gt_2),386).
orientation(mwt1gt_1,160,387). coords(mwt1gt_1,97,126,387).  visible(mwt1gt_1,387). happensAt(walking(mwt1gt_1),387). orientation(mwt1gt_2,140,387). coords(mwt1gt_2,98,109,387).  visible(mwt1gt_2,387). happensAt(walking(mwt1gt_2),387).
orientation(mwt1gt_1,160,388). coords(mwt1gt_1,96,126,388).  visible(mwt1gt_1,388). happensAt(walking(mwt1gt_1),388). orientation(mwt1gt_2,140,388). coords(mwt1gt_2,97,107,388).  visible(mwt1gt_2,388). happensAt(walking(mwt1gt_2),388).
orientation(mwt1gt_1,160,389). coords(mwt1gt_1,96,126,389).  visible(mwt1gt_1,389). happensAt(walking(mwt1gt_1),389). orientation(mwt1gt_2,140,389). coords(mwt1gt_2,97,107,389).  visible(mwt1gt_2,389). happensAt(walking(mwt1gt_2),389).
orientation(mwt1gt_1,160,390). coords(mwt1gt_1,96,126,390).  visible(mwt1gt_1,390). happensAt(walking(mwt1gt_1),390). orientation(mwt1gt_2,140,390). coords(mwt1gt_2,97,107,390).  visible(mwt1gt_2,390). happensAt(walking(mwt1gt_2),390).
orientation(mwt1gt_1,160,391). coords(mwt1gt_1,96,126,391).  visible(mwt1gt_1,391). happensAt(walking(mwt1gt_1),391). orientation(mwt1gt_2,140,391). coords(mwt1gt_2,97,107,391).  visible(mwt1gt_2,391). happensAt(walking(mwt1gt_2),391).
orientation(mwt1gt_1,160,392). coords(mwt1gt_1,96,126,392).  visible(mwt1gt_1,392). happensAt(walking(mwt1gt_1),392). orientation(mwt1gt_2,140,392). coords(mwt1gt_2,96,107,392).  visible(mwt1gt_2,392). happensAt(walking(mwt1gt_2),392).
orientation(mwt1gt_1,160,393). coords(mwt1gt_1,96,126,393).  visible(mwt1gt_1,393). happensAt(walking(mwt1gt_1),393). orientation(mwt1gt_2,140,393). coords(mwt1gt_2,96,107,393).  visible(mwt1gt_2,393). happensAt(walking(mwt1gt_2),393).
orientation(mwt1gt_1,160,394). coords(mwt1gt_1,96,126,394).  visible(mwt1gt_1,394). happensAt(walking(mwt1gt_1),394). orientation(mwt1gt_2,140,394). coords(mwt1gt_2,96,107,394).  visible(mwt1gt_2,394). happensAt(walking(mwt1gt_2),394).
orientation(mwt1gt_1,160,395). coords(mwt1gt_1,96,126,395).  visible(mwt1gt_1,395). happensAt(walking(mwt1gt_1),395). orientation(mwt1gt_2,140,395). coords(mwt1gt_2,96,107,395).  visible(mwt1gt_2,395). happensAt(walking(mwt1gt_2),395).
orientation(mwt1gt_1,160,396). coords(mwt1gt_1,91,120,396).  visible(mwt1gt_1,396). happensAt(walking(mwt1gt_1),396). orientation(mwt1gt_2,140,396). coords(mwt1gt_2,93,105,396).  visible(mwt1gt_2,396). happensAt(walking(mwt1gt_2),396).
orientation(mwt1gt_1,160,397). coords(mwt1gt_1,91,120,397).  visible(mwt1gt_1,397). happensAt(walking(mwt1gt_1),397). orientation(mwt1gt_2,140,397). coords(mwt1gt_2,93,105,397).  visible(mwt1gt_2,397). happensAt(walking(mwt1gt_2),397).
orientation(mwt1gt_1,160,398). coords(mwt1gt_1,91,120,398).  visible(mwt1gt_1,398). happensAt(walking(mwt1gt_1),398). orientation(mwt1gt_2,140,398). coords(mwt1gt_2,93,105,398).  visible(mwt1gt_2,398). happensAt(walking(mwt1gt_2),398).
orientation(mwt1gt_1,160,399). coords(mwt1gt_1,91,120,399).  visible(mwt1gt_1,399). happensAt(walking(mwt1gt_1),399). orientation(mwt1gt_2,140,399). coords(mwt1gt_2,93,105,399).  visible(mwt1gt_2,399). happensAt(walking(mwt1gt_2),399).
orientation(mwt1gt_1,160,400). coords(mwt1gt_1,90,118,400).  visible(mwt1gt_1,400). happensAt(walking(mwt1gt_1),400). orientation(mwt1gt_2,140,400). coords(mwt1gt_2,90,104,400).  visible(mwt1gt_2,400). happensAt(walking(mwt1gt_2),400).
orientation(mwt1gt_1,160,401). coords(mwt1gt_1,90,118,401).  visible(mwt1gt_1,401). happensAt(walking(mwt1gt_1),401). orientation(mwt1gt_2,140,401). coords(mwt1gt_2,90,104,401).  visible(mwt1gt_2,401). happensAt(walking(mwt1gt_2),401).
#show holdsAt/2.