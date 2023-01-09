
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

	orientation(fcgt_2,142,378). coords(fcgt_2,84,93,378).  visible(fcgt_2,378). happensAt(running(fcgt_2),378). orientation(fcgt_3,164,378). coords(fcgt_3,99,187,378).  visible(fcgt_3,378). happensAt(inactive(fcgt_3),378).
orientation(fcgt_2,142,379). coords(fcgt_2,84,93,379).  visible(fcgt_2,379). happensAt(running(fcgt_2),379). orientation(fcgt_3,164,379). coords(fcgt_3,99,187,379).  visible(fcgt_3,379). happensAt(walking(fcgt_3),379).
orientation(fcgt_2,142,380). coords(fcgt_2,84,93,380).  visible(fcgt_2,380). happensAt(running(fcgt_2),380). orientation(fcgt_3,164,380). coords(fcgt_3,99,187,380).  visible(fcgt_3,380). happensAt(walking(fcgt_3),380).
orientation(fcgt_2,142,381). coords(fcgt_2,84,93,381).  visible(fcgt_2,381). happensAt(running(fcgt_2),381). orientation(fcgt_3,164,381). coords(fcgt_3,99,187,381).  visible(fcgt_3,381). happensAt(walking(fcgt_3),381).
orientation(fcgt_2,142,382). coords(fcgt_2,83,92,382).  visible(fcgt_2,382). happensAt(running(fcgt_2),382). orientation(fcgt_3,164,382). coords(fcgt_3,99,187,382).  visible(fcgt_3,382). happensAt(walking(fcgt_3),382).
orientation(fcgt_2,142,383). coords(fcgt_2,83,92,383).  visible(fcgt_2,383). happensAt(running(fcgt_2),383). orientation(fcgt_3,164,383). coords(fcgt_3,99,187,383).  visible(fcgt_3,383). happensAt(walking(fcgt_3),383).
orientation(fcgt_2,142,384). coords(fcgt_2,82,91,384).  visible(fcgt_2,384). happensAt(running(fcgt_2),384). orientation(fcgt_3,164,384). coords(fcgt_3,99,187,384).  visible(fcgt_3,384). happensAt(walking(fcgt_3),384).
orientation(fcgt_2,142,385). coords(fcgt_2,81,90,385).  visible(fcgt_2,385). happensAt(running(fcgt_2),385). orientation(fcgt_3,164,385). coords(fcgt_3,99,188,385).  visible(fcgt_3,385). happensAt(walking(fcgt_3),385).
orientation(fcgt_2,142,386). coords(fcgt_2,81,89,386).  visible(fcgt_2,386). happensAt(running(fcgt_2),386). orientation(fcgt_3,164,386). coords(fcgt_3,99,188,386).  visible(fcgt_3,386). happensAt(walking(fcgt_3),386).
orientation(fcgt_2,142,387). coords(fcgt_2,80,89,387).  visible(fcgt_2,387). happensAt(running(fcgt_2),387). orientation(fcgt_3,164,387). coords(fcgt_3,99,188,387).  visible(fcgt_3,387). happensAt(walking(fcgt_3),387).
orientation(fcgt_2,142,388). coords(fcgt_2,79,88,388).  visible(fcgt_2,388). happensAt(running(fcgt_2),388). orientation(fcgt_3,164,388). coords(fcgt_3,99,189,388).  visible(fcgt_3,388). happensAt(walking(fcgt_3),388).
orientation(fcgt_2,142,389). coords(fcgt_2,79,87,389).  visible(fcgt_2,389). happensAt(running(fcgt_2),389). orientation(fcgt_3,164,389). coords(fcgt_3,99,189,389).  visible(fcgt_3,389). happensAt(walking(fcgt_3),389).
orientation(fcgt_2,139,390). coords(fcgt_2,79,87,390).  visible(fcgt_2,390). happensAt(running(fcgt_2),390). orientation(fcgt_3,164,390). coords(fcgt_3,100,190,390).  visible(fcgt_3,390). happensAt(walking(fcgt_3),390).
orientation(fcgt_2,137,391). coords(fcgt_2,79,86,391).  visible(fcgt_2,391). happensAt(running(fcgt_2),391). orientation(fcgt_3,164,391). coords(fcgt_3,100,190,391).  visible(fcgt_3,391). happensAt(walking(fcgt_3),391).
orientation(fcgt_2,134,392). coords(fcgt_2,78,86,392).  visible(fcgt_2,392). happensAt(running(fcgt_2),392). orientation(fcgt_3,164,392). coords(fcgt_3,100,190,392).  visible(fcgt_3,392). happensAt(walking(fcgt_3),392).
orientation(fcgt_2,130,393). coords(fcgt_2,78,85,393).  visible(fcgt_2,393). happensAt(running(fcgt_2),393). orientation(fcgt_3,164,393). coords(fcgt_3,100,190,393).  visible(fcgt_3,393). happensAt(walking(fcgt_3),393).
orientation(fcgt_2,130,394). coords(fcgt_2,78,84,394).  visible(fcgt_2,394). happensAt(running(fcgt_2),394). orientation(fcgt_3,164,394). coords(fcgt_3,100,191,394).  visible(fcgt_3,394). happensAt(walking(fcgt_3),394).
orientation(fcgt_2,130,395). coords(fcgt_2,78,82,395).  visible(fcgt_2,395). happensAt(running(fcgt_2),395). orientation(fcgt_3,164,395). coords(fcgt_3,100,192,395).  visible(fcgt_3,395). happensAt(walking(fcgt_3),395).
orientation(fcgt_2,130,396). coords(fcgt_2,78,81,396).  visible(fcgt_2,396). happensAt(running(fcgt_2),396). orientation(fcgt_3,164,396). coords(fcgt_3,100,193,396).  visible(fcgt_3,396). happensAt(walking(fcgt_3),396).
orientation(fcgt_2,130,397). coords(fcgt_2,78,81,397).  visible(fcgt_2,397). happensAt(running(fcgt_2),397). orientation(fcgt_3,164,397). coords(fcgt_3,100,194,397).  visible(fcgt_3,397). happensAt(walking(fcgt_3),397).
orientation(fcgt_2,130,398). coords(fcgt_2,77,79,398).  visible(fcgt_2,398). happensAt(running(fcgt_2),398). orientation(fcgt_3,164,398). coords(fcgt_3,100,195,398).  visible(fcgt_3,398). happensAt(walking(fcgt_3),398).
orientation(fcgt_2,130,399). coords(fcgt_2,77,78,399).  visible(fcgt_2,399). happensAt(running(fcgt_2),399). orientation(fcgt_3,164,399). coords(fcgt_3,100,195,399).  visible(fcgt_3,399). happensAt(walking(fcgt_3),399).
orientation(fcgt_2,130,400). coords(fcgt_2,77,77,400).  visible(fcgt_2,400). happensAt(running(fcgt_2),400). orientation(fcgt_3,164,400). coords(fcgt_3,100,196,400).  visible(fcgt_3,400). happensAt(walking(fcgt_3),400).
orientation(fcgt_2,130,401). coords(fcgt_2,77,75,401).  visible(fcgt_2,401). happensAt(running(fcgt_2),401). orientation(fcgt_3,164,401). coords(fcgt_3,100,197,401).  visible(fcgt_3,401). happensAt(walking(fcgt_3),401).
orientation(fcgt_2,130,402). coords(fcgt_2,77,74,402).  visible(fcgt_2,402). happensAt(running(fcgt_2),402). orientation(fcgt_3,164,402). coords(fcgt_3,99,197,402).  visible(fcgt_3,402). happensAt(walking(fcgt_3),402).
#show holdsAt/2.