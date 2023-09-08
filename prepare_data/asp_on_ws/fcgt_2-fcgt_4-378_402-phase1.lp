
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

	orientation(fcgt_2,142,378). coords(fcgt_2,84,93,378).  visible(fcgt_2,378). happensAt(running(fcgt_2),378). orientation(fcgt_4,166,378). coords(fcgt_4,82,206,378).  visible(fcgt_4,378). happensAt(inactive(fcgt_4),378).
orientation(fcgt_2,142,379). coords(fcgt_2,84,93,379).  visible(fcgt_2,379). happensAt(running(fcgt_2),379). orientation(fcgt_4,166,379). coords(fcgt_4,82,206,379).  visible(fcgt_4,379). happensAt(inactive(fcgt_4),379).
orientation(fcgt_2,142,380). coords(fcgt_2,84,93,380).  visible(fcgt_2,380). happensAt(running(fcgt_2),380). orientation(fcgt_4,166,380). coords(fcgt_4,82,206,380).  visible(fcgt_4,380). happensAt(inactive(fcgt_4),380).
orientation(fcgt_2,142,381). coords(fcgt_2,84,93,381).  visible(fcgt_2,381). happensAt(running(fcgt_2),381). orientation(fcgt_4,166,381). coords(fcgt_4,82,206,381).  visible(fcgt_4,381). happensAt(inactive(fcgt_4),381).
orientation(fcgt_2,142,382). coords(fcgt_2,83,92,382).  visible(fcgt_2,382). happensAt(running(fcgt_2),382). orientation(fcgt_4,166,382). coords(fcgt_4,82,206,382).  visible(fcgt_4,382). happensAt(inactive(fcgt_4),382).
orientation(fcgt_2,142,383). coords(fcgt_2,83,92,383).  visible(fcgt_2,383). happensAt(running(fcgt_2),383). orientation(fcgt_4,166,383). coords(fcgt_4,82,206,383).  visible(fcgt_4,383). happensAt(inactive(fcgt_4),383).
orientation(fcgt_2,142,384). coords(fcgt_2,82,91,384).  visible(fcgt_2,384). happensAt(running(fcgt_2),384). orientation(fcgt_4,166,384). coords(fcgt_4,82,206,384).  visible(fcgt_4,384). happensAt(inactive(fcgt_4),384).
orientation(fcgt_2,142,385). coords(fcgt_2,81,90,385).  visible(fcgt_2,385). happensAt(running(fcgt_2),385). orientation(fcgt_4,166,385). coords(fcgt_4,82,206,385).  visible(fcgt_4,385). happensAt(inactive(fcgt_4),385).
orientation(fcgt_2,142,386). coords(fcgt_2,81,89,386).  visible(fcgt_2,386). happensAt(running(fcgt_2),386). orientation(fcgt_4,166,386). coords(fcgt_4,82,206,386).  visible(fcgt_4,386). happensAt(inactive(fcgt_4),386).
orientation(fcgt_2,142,387). coords(fcgt_2,80,89,387).  visible(fcgt_2,387). happensAt(running(fcgt_2),387). orientation(fcgt_4,166,387). coords(fcgt_4,82,206,387).  visible(fcgt_4,387). happensAt(inactive(fcgt_4),387).
orientation(fcgt_2,142,388). coords(fcgt_2,79,88,388).  visible(fcgt_2,388). happensAt(running(fcgt_2),388). orientation(fcgt_4,166,388). coords(fcgt_4,82,206,388).  visible(fcgt_4,388). happensAt(inactive(fcgt_4),388).
orientation(fcgt_2,142,389). coords(fcgt_2,79,87,389).  visible(fcgt_2,389). happensAt(running(fcgt_2),389). orientation(fcgt_4,166,389). coords(fcgt_4,82,206,389).  visible(fcgt_4,389). happensAt(inactive(fcgt_4),389).
orientation(fcgt_2,139,390). coords(fcgt_2,79,87,390).  visible(fcgt_2,390). happensAt(running(fcgt_2),390). orientation(fcgt_4,166,390). coords(fcgt_4,82,206,390).  visible(fcgt_4,390). happensAt(inactive(fcgt_4),390).
orientation(fcgt_2,137,391). coords(fcgt_2,79,86,391).  visible(fcgt_2,391). happensAt(running(fcgt_2),391). orientation(fcgt_4,166,391). coords(fcgt_4,82,206,391).  visible(fcgt_4,391). happensAt(inactive(fcgt_4),391).
orientation(fcgt_2,134,392). coords(fcgt_2,78,86,392).  visible(fcgt_2,392). happensAt(running(fcgt_2),392). orientation(fcgt_4,166,392). coords(fcgt_4,82,206,392).  visible(fcgt_4,392). happensAt(inactive(fcgt_4),392).
orientation(fcgt_2,130,393). coords(fcgt_2,78,85,393).  visible(fcgt_2,393). happensAt(running(fcgt_2),393). orientation(fcgt_4,166,393). coords(fcgt_4,82,206,393).  visible(fcgt_4,393). happensAt(inactive(fcgt_4),393).
orientation(fcgt_2,130,394). coords(fcgt_2,78,84,394).  visible(fcgt_2,394). happensAt(running(fcgt_2),394). orientation(fcgt_4,166,394). coords(fcgt_4,82,206,394).  visible(fcgt_4,394). happensAt(inactive(fcgt_4),394).
orientation(fcgt_2,130,395). coords(fcgt_2,78,82,395).  visible(fcgt_2,395). happensAt(running(fcgt_2),395). orientation(fcgt_4,166,395). coords(fcgt_4,82,206,395).  visible(fcgt_4,395). happensAt(inactive(fcgt_4),395).
orientation(fcgt_2,130,396). coords(fcgt_2,78,81,396).  visible(fcgt_2,396). happensAt(running(fcgt_2),396). orientation(fcgt_4,166,396). coords(fcgt_4,82,206,396).  visible(fcgt_4,396). happensAt(inactive(fcgt_4),396).
orientation(fcgt_2,130,397). coords(fcgt_2,78,81,397).  visible(fcgt_2,397). happensAt(running(fcgt_2),397). orientation(fcgt_4,166,397). coords(fcgt_4,82,206,397).  visible(fcgt_4,397). happensAt(inactive(fcgt_4),397).
orientation(fcgt_2,130,398). coords(fcgt_2,77,79,398).  visible(fcgt_2,398). happensAt(running(fcgt_2),398). orientation(fcgt_4,166,398). coords(fcgt_4,82,206,398).  visible(fcgt_4,398). happensAt(inactive(fcgt_4),398).
orientation(fcgt_2,130,399). coords(fcgt_2,77,78,399).  visible(fcgt_2,399). happensAt(running(fcgt_2),399). orientation(fcgt_4,166,399). coords(fcgt_4,82,206,399).  visible(fcgt_4,399). happensAt(inactive(fcgt_4),399).
orientation(fcgt_2,130,400). coords(fcgt_2,77,77,400).  visible(fcgt_2,400). happensAt(running(fcgt_2),400). orientation(fcgt_4,166,400). coords(fcgt_4,82,206,400).  visible(fcgt_4,400). happensAt(inactive(fcgt_4),400).
orientation(fcgt_2,130,401). coords(fcgt_2,77,75,401).  visible(fcgt_2,401). happensAt(running(fcgt_2),401). orientation(fcgt_4,166,401). coords(fcgt_4,82,206,401).  visible(fcgt_4,401). happensAt(inactive(fcgt_4),401).
orientation(fcgt_2,130,402). coords(fcgt_2,77,74,402).  visible(fcgt_2,402). happensAt(running(fcgt_2),402). orientation(fcgt_4,166,402). coords(fcgt_4,82,206,402).  visible(fcgt_4,402). happensAt(inactive(fcgt_4),402).
#show holdsAt/2.