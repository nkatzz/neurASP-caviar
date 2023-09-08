
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

	orientation(fomdgt3_6,140,386). coords(fomdgt3_6,102,88,386).  visible(fomdgt3_6,386). happensAt(running(fomdgt3_6),386). orientation(fomdgt3_7,134,386). coords(fomdgt3_7,77,93,386).  visible(fomdgt3_7,386). happensAt(walking(fomdgt3_7),386).
orientation(fomdgt3_6,140,387). coords(fomdgt3_6,100,88,387).  visible(fomdgt3_6,387). happensAt(running(fomdgt3_6),387). orientation(fomdgt3_7,134,387). coords(fomdgt3_7,77,93,387).  visible(fomdgt3_7,387). happensAt(walking(fomdgt3_7),387).
orientation(fomdgt3_6,140,388). coords(fomdgt3_6,100,88,388).  visible(fomdgt3_6,388). happensAt(running(fomdgt3_6),388). orientation(fomdgt3_7,134,388). coords(fomdgt3_7,77,93,388).  visible(fomdgt3_7,388). happensAt(walking(fomdgt3_7),388).
orientation(fomdgt3_6,140,389). coords(fomdgt3_6,99,88,389).  visible(fomdgt3_6,389). happensAt(running(fomdgt3_6),389). orientation(fomdgt3_7,134,389). coords(fomdgt3_7,77,93,389).  visible(fomdgt3_7,389). happensAt(walking(fomdgt3_7),389).
orientation(fomdgt3_6,140,390). coords(fomdgt3_6,96,85,390).  visible(fomdgt3_6,390). happensAt(running(fomdgt3_6),390). orientation(fomdgt3_7,134,390). coords(fomdgt3_7,77,93,390).  visible(fomdgt3_7,390). happensAt(walking(fomdgt3_7),390).
orientation(fomdgt3_6,140,391). coords(fomdgt3_6,93,84,391).  visible(fomdgt3_6,391). happensAt(running(fomdgt3_6),391). orientation(fomdgt3_7,134,391). coords(fomdgt3_7,77,94,391).  visible(fomdgt3_7,391). happensAt(walking(fomdgt3_7),391).
orientation(fomdgt3_6,140,392). coords(fomdgt3_6,92,83,392).  visible(fomdgt3_6,392). happensAt(running(fomdgt3_6),392). orientation(fomdgt3_7,134,392). coords(fomdgt3_7,78,96,392).  visible(fomdgt3_7,392). happensAt(walking(fomdgt3_7),392).
orientation(fomdgt3_6,140,393). coords(fomdgt3_6,92,83,393).  visible(fomdgt3_6,393). happensAt(running(fomdgt3_6),393). orientation(fomdgt3_7,134,393). coords(fomdgt3_7,78,97,393).  visible(fomdgt3_7,393). happensAt(walking(fomdgt3_7),393).
orientation(fomdgt3_6,140,394). coords(fomdgt3_6,91,83,394).  visible(fomdgt3_6,394). happensAt(running(fomdgt3_6),394). orientation(fomdgt3_7,134,394). coords(fomdgt3_7,78,97,394).  visible(fomdgt3_7,394). happensAt(walking(fomdgt3_7),394).
orientation(fomdgt3_6,140,395). coords(fomdgt3_6,90,82,395).  visible(fomdgt3_6,395). happensAt(running(fomdgt3_6),395). orientation(fomdgt3_7,134,395). coords(fomdgt3_7,79,97,395).  visible(fomdgt3_7,395). happensAt(walking(fomdgt3_7),395).
orientation(fomdgt3_6,140,396). coords(fomdgt3_6,88,82,396).  visible(fomdgt3_6,396). happensAt(running(fomdgt3_6),396). orientation(fomdgt3_7,134,396). coords(fomdgt3_7,79,98,396).  visible(fomdgt3_7,396). happensAt(walking(fomdgt3_7),396).
orientation(fomdgt3_6,140,397). coords(fomdgt3_6,87,81,397).  visible(fomdgt3_6,397). happensAt(running(fomdgt3_6),397). orientation(fomdgt3_7,134,397). coords(fomdgt3_7,79,98,397).  visible(fomdgt3_7,397). happensAt(walking(fomdgt3_7),397).
orientation(fomdgt3_6,140,398). coords(fomdgt3_6,85,81,398).  visible(fomdgt3_6,398). happensAt(running(fomdgt3_6),398). orientation(fomdgt3_7,134,398). coords(fomdgt3_7,79,99,398).  visible(fomdgt3_7,398). happensAt(walking(fomdgt3_7),398).
orientation(fomdgt3_6,140,399). coords(fomdgt3_6,83,81,399).  visible(fomdgt3_6,399). happensAt(running(fomdgt3_6),399). orientation(fomdgt3_7,134,399). coords(fomdgt3_7,79,99,399).  visible(fomdgt3_7,399). happensAt(walking(fomdgt3_7),399).
orientation(fomdgt3_6,140,400). coords(fomdgt3_6,82,81,400).  visible(fomdgt3_6,400). happensAt(running(fomdgt3_6),400). orientation(fomdgt3_7,134,400). coords(fomdgt3_7,79,101,400).  visible(fomdgt3_7,400). happensAt(walking(fomdgt3_7),400).
orientation(fomdgt3_6,140,401). coords(fomdgt3_6,79,81,401).  visible(fomdgt3_6,401). happensAt(running(fomdgt3_6),401). orientation(fomdgt3_7,134,401). coords(fomdgt3_7,79,101,401).  visible(fomdgt3_7,401). happensAt(walking(fomdgt3_7),401).
orientation(fomdgt3_6,140,402). coords(fomdgt3_6,79,81,402).  visible(fomdgt3_6,402). happensAt(running(fomdgt3_6),402). orientation(fomdgt3_7,134,402). coords(fomdgt3_7,80,101,402).  visible(fomdgt3_7,402). happensAt(walking(fomdgt3_7),402).
orientation(fomdgt3_6,140,403). coords(fomdgt3_6,77,81,403).  visible(fomdgt3_6,403). happensAt(running(fomdgt3_6),403). orientation(fomdgt3_7,134,403). coords(fomdgt3_7,80,103,403).  visible(fomdgt3_7,403). happensAt(walking(fomdgt3_7),403).
orientation(fomdgt3_6,140,404). coords(fomdgt3_6,76,81,404).  visible(fomdgt3_6,404). happensAt(running(fomdgt3_6),404). orientation(fomdgt3_7,134,404). coords(fomdgt3_7,80,103,404).  visible(fomdgt3_7,404). happensAt(walking(fomdgt3_7),404).
orientation(fomdgt3_6,140,405). coords(fomdgt3_6,75,81,405).  visible(fomdgt3_6,405). happensAt(running(fomdgt3_6),405). orientation(fomdgt3_7,134,405). coords(fomdgt3_7,80,103,405).  visible(fomdgt3_7,405). happensAt(walking(fomdgt3_7),405).
orientation(fomdgt3_6,140,406). coords(fomdgt3_6,74,81,406).  visible(fomdgt3_6,406). happensAt(running(fomdgt3_6),406). orientation(fomdgt3_7,134,406). coords(fomdgt3_7,80,103,406).  visible(fomdgt3_7,406). happensAt(walking(fomdgt3_7),406).
orientation(fomdgt3_6,140,407). coords(fomdgt3_6,71,82,407).  visible(fomdgt3_6,407). happensAt(running(fomdgt3_6),407). orientation(fomdgt3_7,134,407). coords(fomdgt3_7,80,104,407).  visible(fomdgt3_7,407). happensAt(walking(fomdgt3_7),407).
orientation(fomdgt3_6,140,408). coords(fomdgt3_6,71,82,408).  visible(fomdgt3_6,408). happensAt(running(fomdgt3_6),408). orientation(fomdgt3_7,134,408). coords(fomdgt3_7,80,106,408).  visible(fomdgt3_7,408). happensAt(walking(fomdgt3_7),408).
orientation(fomdgt3_6,140,409). coords(fomdgt3_6,69,82,409).  visible(fomdgt3_6,409). happensAt(running(fomdgt3_6),409). orientation(fomdgt3_7,134,409). coords(fomdgt3_7,80,108,409).  visible(fomdgt3_7,409). happensAt(walking(fomdgt3_7),409).
orientation(fomdgt3_6,140,410). coords(fomdgt3_6,68,82,410).  visible(fomdgt3_6,410). happensAt(running(fomdgt3_6),410). orientation(fomdgt3_7,134,410). coords(fomdgt3_7,80,109,410).  visible(fomdgt3_7,410). happensAt(walking(fomdgt3_7),410).
#show holdsAt/2.