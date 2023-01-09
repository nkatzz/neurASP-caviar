
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
orientation(fra1gt_6,160,391). coords(fra1gt_6,139,98,391).  visible(fra1gt_6,391). happensAt(running(fra1gt_6),391). orientation(fra1gt_7,162,391). coords(fra1gt_7,126,75,391).  visible(fra1gt_7,391). happensAt(running(fra1gt_7),391).
orientation(fra1gt_6,160,392). coords(fra1gt_6,138,98,392).  visible(fra1gt_6,392). happensAt(running(fra1gt_6),392). orientation(fra1gt_7,162,392). coords(fra1gt_7,123,75,392).  visible(fra1gt_7,392). happensAt(running(fra1gt_7),392).
orientation(fra1gt_6,160,393). coords(fra1gt_6,135,98,393).  visible(fra1gt_6,393). happensAt(running(fra1gt_6),393). orientation(fra1gt_7,162,393). coords(fra1gt_7,120,75,393).  visible(fra1gt_7,393). happensAt(running(fra1gt_7),393).
orientation(fra1gt_6,160,394). coords(fra1gt_6,135,99,394).  visible(fra1gt_6,394). happensAt(running(fra1gt_6),394). orientation(fra1gt_7,162,394). coords(fra1gt_7,117,75,394).  visible(fra1gt_7,394). happensAt(running(fra1gt_7),394).
orientation(fra1gt_6,160,395). coords(fra1gt_6,130,101,395).  visible(fra1gt_6,395). happensAt(running(fra1gt_6),395). orientation(fra1gt_7,162,395). coords(fra1gt_7,115,75,395).  visible(fra1gt_7,395). happensAt(running(fra1gt_7),395).
orientation(fra1gt_6,160,396). coords(fra1gt_6,127,102,396).  visible(fra1gt_6,396). happensAt(running(fra1gt_6),396). orientation(fra1gt_7,162,396). coords(fra1gt_7,113,76,396).  visible(fra1gt_7,396). happensAt(running(fra1gt_7),396).
orientation(fra1gt_6,160,397). coords(fra1gt_6,127,104,397).  visible(fra1gt_6,397). happensAt(running(fra1gt_6),397). orientation(fra1gt_7,162,397). coords(fra1gt_7,112,75,397).  visible(fra1gt_7,397). happensAt(running(fra1gt_7),397).
orientation(fra1gt_6,160,398). coords(fra1gt_6,125,106,398).  visible(fra1gt_6,398). happensAt(running(fra1gt_6),398). orientation(fra1gt_7,162,398). coords(fra1gt_7,110,75,398).  visible(fra1gt_7,398). happensAt(running(fra1gt_7),398).
orientation(fra1gt_6,160,399). coords(fra1gt_6,123,108,399).  visible(fra1gt_6,399). happensAt(running(fra1gt_6),399). orientation(fra1gt_7,158,399). coords(fra1gt_7,108,76,399).  visible(fra1gt_7,399). happensAt(running(fra1gt_7),399).
orientation(fra1gt_6,160,400). coords(fra1gt_6,122,110,400).  visible(fra1gt_6,400). happensAt(running(fra1gt_6),400). orientation(fra1gt_7,155,400). coords(fra1gt_7,105,76,400).  visible(fra1gt_7,400). happensAt(running(fra1gt_7),400).
orientation(fra1gt_6,160,401). coords(fra1gt_6,119,112,401).  visible(fra1gt_6,401). happensAt(running(fra1gt_6),401). orientation(fra1gt_7,146,401). coords(fra1gt_7,103,77,401).  visible(fra1gt_7,401). happensAt(running(fra1gt_7),401).
orientation(fra1gt_6,160,402). coords(fra1gt_6,117,113,402).  visible(fra1gt_6,402). happensAt(running(fra1gt_6),402). orientation(fra1gt_7,146,402). coords(fra1gt_7,102,78,402).  visible(fra1gt_7,402). happensAt(running(fra1gt_7),402).
#show holdsAt/2.