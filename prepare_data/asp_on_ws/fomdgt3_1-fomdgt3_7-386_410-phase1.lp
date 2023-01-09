
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

	orientation(fomdgt3_1,150,386). coords(fomdgt3_1,46,102,386).  visible(fomdgt3_1,386). happensAt(inactive(fomdgt3_1),386). orientation(fomdgt3_7,134,386). coords(fomdgt3_7,77,93,386).  visible(fomdgt3_7,386). happensAt(walking(fomdgt3_7),386).
orientation(fomdgt3_1,150,387). coords(fomdgt3_1,46,102,387).  visible(fomdgt3_1,387). happensAt(inactive(fomdgt3_1),387). orientation(fomdgt3_7,134,387). coords(fomdgt3_7,77,93,387).  visible(fomdgt3_7,387). happensAt(walking(fomdgt3_7),387).
orientation(fomdgt3_1,150,388). coords(fomdgt3_1,46,102,388).  visible(fomdgt3_1,388). happensAt(inactive(fomdgt3_1),388). orientation(fomdgt3_7,134,388). coords(fomdgt3_7,77,93,388).  visible(fomdgt3_7,388). happensAt(walking(fomdgt3_7),388).
orientation(fomdgt3_1,150,389). coords(fomdgt3_1,46,102,389).  visible(fomdgt3_1,389). happensAt(inactive(fomdgt3_1),389). orientation(fomdgt3_7,134,389). coords(fomdgt3_7,77,93,389).  visible(fomdgt3_7,389). happensAt(walking(fomdgt3_7),389).
orientation(fomdgt3_1,150,390). coords(fomdgt3_1,46,102,390).  visible(fomdgt3_1,390). happensAt(inactive(fomdgt3_1),390). orientation(fomdgt3_7,134,390). coords(fomdgt3_7,77,93,390).  visible(fomdgt3_7,390). happensAt(walking(fomdgt3_7),390).
orientation(fomdgt3_1,150,391). coords(fomdgt3_1,46,102,391).  visible(fomdgt3_1,391). happensAt(inactive(fomdgt3_1),391). orientation(fomdgt3_7,134,391). coords(fomdgt3_7,77,94,391).  visible(fomdgt3_7,391). happensAt(walking(fomdgt3_7),391).
orientation(fomdgt3_1,150,392). coords(fomdgt3_1,46,102,392).  visible(fomdgt3_1,392). happensAt(inactive(fomdgt3_1),392). orientation(fomdgt3_7,134,392). coords(fomdgt3_7,78,96,392).  visible(fomdgt3_7,392). happensAt(walking(fomdgt3_7),392).
orientation(fomdgt3_1,150,393). coords(fomdgt3_1,46,102,393).  visible(fomdgt3_1,393). happensAt(inactive(fomdgt3_1),393). orientation(fomdgt3_7,134,393). coords(fomdgt3_7,78,97,393).  visible(fomdgt3_7,393). happensAt(walking(fomdgt3_7),393).
orientation(fomdgt3_1,150,394). coords(fomdgt3_1,46,102,394).  visible(fomdgt3_1,394). happensAt(inactive(fomdgt3_1),394). orientation(fomdgt3_7,134,394). coords(fomdgt3_7,78,97,394).  visible(fomdgt3_7,394). happensAt(walking(fomdgt3_7),394).
orientation(fomdgt3_1,150,395). coords(fomdgt3_1,46,102,395).  visible(fomdgt3_1,395). happensAt(inactive(fomdgt3_1),395). orientation(fomdgt3_7,134,395). coords(fomdgt3_7,79,97,395).  visible(fomdgt3_7,395). happensAt(walking(fomdgt3_7),395).
orientation(fomdgt3_1,150,396). coords(fomdgt3_1,46,102,396).  visible(fomdgt3_1,396). happensAt(inactive(fomdgt3_1),396). orientation(fomdgt3_7,134,396). coords(fomdgt3_7,79,98,396).  visible(fomdgt3_7,396). happensAt(walking(fomdgt3_7),396).
orientation(fomdgt3_1,150,397). coords(fomdgt3_1,46,102,397).  visible(fomdgt3_1,397). happensAt(inactive(fomdgt3_1),397). orientation(fomdgt3_7,134,397). coords(fomdgt3_7,79,98,397).  visible(fomdgt3_7,397). happensAt(walking(fomdgt3_7),397).
orientation(fomdgt3_1,150,398). coords(fomdgt3_1,46,102,398).  visible(fomdgt3_1,398). happensAt(inactive(fomdgt3_1),398). orientation(fomdgt3_7,134,398). coords(fomdgt3_7,79,99,398).  visible(fomdgt3_7,398). happensAt(walking(fomdgt3_7),398).
orientation(fomdgt3_1,150,399). coords(fomdgt3_1,46,102,399).  visible(fomdgt3_1,399). happensAt(inactive(fomdgt3_1),399). orientation(fomdgt3_7,134,399). coords(fomdgt3_7,79,99,399).  visible(fomdgt3_7,399). happensAt(walking(fomdgt3_7),399).
orientation(fomdgt3_1,150,400). coords(fomdgt3_1,46,102,400).  visible(fomdgt3_1,400). happensAt(inactive(fomdgt3_1),400). orientation(fomdgt3_7,134,400). coords(fomdgt3_7,79,101,400).  visible(fomdgt3_7,400). happensAt(walking(fomdgt3_7),400).
orientation(fomdgt3_1,150,401). coords(fomdgt3_1,46,102,401).  visible(fomdgt3_1,401). happensAt(inactive(fomdgt3_1),401). orientation(fomdgt3_7,134,401). coords(fomdgt3_7,79,101,401).  visible(fomdgt3_7,401). happensAt(walking(fomdgt3_7),401).
orientation(fomdgt3_1,150,402). coords(fomdgt3_1,46,102,402).  visible(fomdgt3_1,402). happensAt(inactive(fomdgt3_1),402). orientation(fomdgt3_7,134,402). coords(fomdgt3_7,80,101,402).  visible(fomdgt3_7,402). happensAt(walking(fomdgt3_7),402).
orientation(fomdgt3_1,150,403). coords(fomdgt3_1,46,102,403).  visible(fomdgt3_1,403). happensAt(inactive(fomdgt3_1),403). orientation(fomdgt3_7,134,403). coords(fomdgt3_7,80,103,403).  visible(fomdgt3_7,403). happensAt(walking(fomdgt3_7),403).
orientation(fomdgt3_1,150,404). coords(fomdgt3_1,46,102,404).  visible(fomdgt3_1,404). happensAt(inactive(fomdgt3_1),404). orientation(fomdgt3_7,134,404). coords(fomdgt3_7,80,103,404).  visible(fomdgt3_7,404). happensAt(walking(fomdgt3_7),404).
orientation(fomdgt3_1,150,405). coords(fomdgt3_1,46,102,405).  visible(fomdgt3_1,405). happensAt(inactive(fomdgt3_1),405). orientation(fomdgt3_7,134,405). coords(fomdgt3_7,80,103,405).  visible(fomdgt3_7,405). happensAt(walking(fomdgt3_7),405).
orientation(fomdgt3_1,150,406). coords(fomdgt3_1,46,102,406).  visible(fomdgt3_1,406). happensAt(inactive(fomdgt3_1),406). orientation(fomdgt3_7,134,406). coords(fomdgt3_7,80,103,406).  visible(fomdgt3_7,406). happensAt(walking(fomdgt3_7),406).
orientation(fomdgt3_1,150,407). coords(fomdgt3_1,46,102,407).  visible(fomdgt3_1,407). happensAt(inactive(fomdgt3_1),407). orientation(fomdgt3_7,134,407). coords(fomdgt3_7,80,104,407).  visible(fomdgt3_7,407). happensAt(walking(fomdgt3_7),407).
orientation(fomdgt3_1,150,408). coords(fomdgt3_1,46,102,408).  visible(fomdgt3_1,408). happensAt(inactive(fomdgt3_1),408). orientation(fomdgt3_7,134,408). coords(fomdgt3_7,80,106,408).  visible(fomdgt3_7,408). happensAt(walking(fomdgt3_7),408).
orientation(fomdgt3_1,150,409). coords(fomdgt3_1,46,102,409).  visible(fomdgt3_1,409). happensAt(inactive(fomdgt3_1),409). orientation(fomdgt3_7,134,409). coords(fomdgt3_7,80,108,409).  visible(fomdgt3_7,409). happensAt(walking(fomdgt3_7),409).
orientation(fomdgt3_1,150,410). coords(fomdgt3_1,46,102,410).  visible(fomdgt3_1,410). happensAt(inactive(fomdgt3_1),410). orientation(fomdgt3_7,134,410). coords(fomdgt3_7,80,109,410).  visible(fomdgt3_7,410). happensAt(walking(fomdgt3_7),410).
#show holdsAt/2.