
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

	orientation(fomdgt3_1,150,378). coords(fomdgt3_1,46,102,378).  visible(fomdgt3_1,378). happensAt(inactive(fomdgt3_1),378). orientation(fomdgt3_6,156,378). coords(fomdgt3_6,118,91,378).  visible(fomdgt3_6,378). happensAt(running(fomdgt3_6),378).
orientation(fomdgt3_1,150,379). coords(fomdgt3_1,46,102,379).  visible(fomdgt3_1,379). happensAt(inactive(fomdgt3_1),379). orientation(fomdgt3_6,156,379). coords(fomdgt3_6,116,91,379).  visible(fomdgt3_6,379). happensAt(running(fomdgt3_6),379).
orientation(fomdgt3_1,150,380). coords(fomdgt3_1,46,102,380).  visible(fomdgt3_1,380). happensAt(inactive(fomdgt3_1),380). orientation(fomdgt3_6,156,380). coords(fomdgt3_6,115,91,380).  visible(fomdgt3_6,380). happensAt(running(fomdgt3_6),380).
orientation(fomdgt3_1,150,381). coords(fomdgt3_1,46,102,381).  visible(fomdgt3_1,381). happensAt(inactive(fomdgt3_1),381). orientation(fomdgt3_6,156,381). coords(fomdgt3_6,114,90,381).  visible(fomdgt3_6,381). happensAt(running(fomdgt3_6),381).
orientation(fomdgt3_1,150,382). coords(fomdgt3_1,46,102,382).  visible(fomdgt3_1,382). happensAt(inactive(fomdgt3_1),382). orientation(fomdgt3_6,156,382). coords(fomdgt3_6,113,90,382).  visible(fomdgt3_6,382). happensAt(running(fomdgt3_6),382).
orientation(fomdgt3_1,150,383). coords(fomdgt3_1,46,102,383).  visible(fomdgt3_1,383). happensAt(inactive(fomdgt3_1),383). orientation(fomdgt3_6,156,383). coords(fomdgt3_6,108,89,383).  visible(fomdgt3_6,383). happensAt(running(fomdgt3_6),383).
orientation(fomdgt3_1,150,384). coords(fomdgt3_1,46,102,384).  visible(fomdgt3_1,384). happensAt(inactive(fomdgt3_1),384). orientation(fomdgt3_6,140,384). coords(fomdgt3_6,106,89,384).  visible(fomdgt3_6,384). happensAt(running(fomdgt3_6),384).
orientation(fomdgt3_1,150,385). coords(fomdgt3_1,46,102,385).  visible(fomdgt3_1,385). happensAt(inactive(fomdgt3_1),385). orientation(fomdgt3_6,140,385). coords(fomdgt3_6,103,88,385).  visible(fomdgt3_6,385). happensAt(running(fomdgt3_6),385).
orientation(fomdgt3_1,150,386). coords(fomdgt3_1,46,102,386).  visible(fomdgt3_1,386). happensAt(inactive(fomdgt3_1),386). orientation(fomdgt3_6,140,386). coords(fomdgt3_6,102,88,386).  visible(fomdgt3_6,386). happensAt(running(fomdgt3_6),386).
orientation(fomdgt3_1,150,387). coords(fomdgt3_1,46,102,387).  visible(fomdgt3_1,387). happensAt(inactive(fomdgt3_1),387). orientation(fomdgt3_6,140,387). coords(fomdgt3_6,100,88,387).  visible(fomdgt3_6,387). happensAt(running(fomdgt3_6),387).
orientation(fomdgt3_1,150,388). coords(fomdgt3_1,46,102,388).  visible(fomdgt3_1,388). happensAt(inactive(fomdgt3_1),388). orientation(fomdgt3_6,140,388). coords(fomdgt3_6,100,88,388).  visible(fomdgt3_6,388). happensAt(running(fomdgt3_6),388).
orientation(fomdgt3_1,150,389). coords(fomdgt3_1,46,102,389).  visible(fomdgt3_1,389). happensAt(inactive(fomdgt3_1),389). orientation(fomdgt3_6,140,389). coords(fomdgt3_6,99,88,389).  visible(fomdgt3_6,389). happensAt(running(fomdgt3_6),389).
orientation(fomdgt3_1,150,390). coords(fomdgt3_1,46,102,390).  visible(fomdgt3_1,390). happensAt(inactive(fomdgt3_1),390). orientation(fomdgt3_6,140,390). coords(fomdgt3_6,96,85,390).  visible(fomdgt3_6,390). happensAt(running(fomdgt3_6),390).
orientation(fomdgt3_1,150,391). coords(fomdgt3_1,46,102,391).  visible(fomdgt3_1,391). happensAt(inactive(fomdgt3_1),391). orientation(fomdgt3_6,140,391). coords(fomdgt3_6,93,84,391).  visible(fomdgt3_6,391). happensAt(running(fomdgt3_6),391).
orientation(fomdgt3_1,150,392). coords(fomdgt3_1,46,102,392).  visible(fomdgt3_1,392). happensAt(inactive(fomdgt3_1),392). orientation(fomdgt3_6,140,392). coords(fomdgt3_6,92,83,392).  visible(fomdgt3_6,392). happensAt(running(fomdgt3_6),392).
orientation(fomdgt3_1,150,393). coords(fomdgt3_1,46,102,393).  visible(fomdgt3_1,393). happensAt(inactive(fomdgt3_1),393). orientation(fomdgt3_6,140,393). coords(fomdgt3_6,92,83,393).  visible(fomdgt3_6,393). happensAt(running(fomdgt3_6),393).
orientation(fomdgt3_1,150,394). coords(fomdgt3_1,46,102,394).  visible(fomdgt3_1,394). happensAt(inactive(fomdgt3_1),394). orientation(fomdgt3_6,140,394). coords(fomdgt3_6,91,83,394).  visible(fomdgt3_6,394). happensAt(running(fomdgt3_6),394).
orientation(fomdgt3_1,150,395). coords(fomdgt3_1,46,102,395).  visible(fomdgt3_1,395). happensAt(inactive(fomdgt3_1),395). orientation(fomdgt3_6,140,395). coords(fomdgt3_6,90,82,395).  visible(fomdgt3_6,395). happensAt(running(fomdgt3_6),395).
orientation(fomdgt3_1,150,396). coords(fomdgt3_1,46,102,396).  visible(fomdgt3_1,396). happensAt(inactive(fomdgt3_1),396). orientation(fomdgt3_6,140,396). coords(fomdgt3_6,88,82,396).  visible(fomdgt3_6,396). happensAt(running(fomdgt3_6),396).
orientation(fomdgt3_1,150,397). coords(fomdgt3_1,46,102,397).  visible(fomdgt3_1,397). happensAt(inactive(fomdgt3_1),397). orientation(fomdgt3_6,140,397). coords(fomdgt3_6,87,81,397).  visible(fomdgt3_6,397). happensAt(running(fomdgt3_6),397).
orientation(fomdgt3_1,150,398). coords(fomdgt3_1,46,102,398).  visible(fomdgt3_1,398). happensAt(inactive(fomdgt3_1),398). orientation(fomdgt3_6,140,398). coords(fomdgt3_6,85,81,398).  visible(fomdgt3_6,398). happensAt(running(fomdgt3_6),398).
orientation(fomdgt3_1,150,399). coords(fomdgt3_1,46,102,399).  visible(fomdgt3_1,399). happensAt(inactive(fomdgt3_1),399). orientation(fomdgt3_6,140,399). coords(fomdgt3_6,83,81,399).  visible(fomdgt3_6,399). happensAt(running(fomdgt3_6),399).
orientation(fomdgt3_1,150,400). coords(fomdgt3_1,46,102,400).  visible(fomdgt3_1,400). happensAt(inactive(fomdgt3_1),400). orientation(fomdgt3_6,140,400). coords(fomdgt3_6,82,81,400).  visible(fomdgt3_6,400). happensAt(running(fomdgt3_6),400).
orientation(fomdgt3_1,150,401). coords(fomdgt3_1,46,102,401).  visible(fomdgt3_1,401). happensAt(inactive(fomdgt3_1),401). orientation(fomdgt3_6,140,401). coords(fomdgt3_6,79,81,401).  visible(fomdgt3_6,401). happensAt(running(fomdgt3_6),401).
orientation(fomdgt3_1,150,402). coords(fomdgt3_1,46,102,402).  visible(fomdgt3_1,402). happensAt(inactive(fomdgt3_1),402). orientation(fomdgt3_6,140,402). coords(fomdgt3_6,79,81,402).  visible(fomdgt3_6,402). happensAt(running(fomdgt3_6),402).
#show holdsAt/2.