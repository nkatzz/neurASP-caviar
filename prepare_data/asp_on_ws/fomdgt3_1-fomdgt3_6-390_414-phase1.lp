
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
orientation(fomdgt3_1,150,403). coords(fomdgt3_1,46,102,403).  visible(fomdgt3_1,403). happensAt(inactive(fomdgt3_1),403). orientation(fomdgt3_6,140,403). coords(fomdgt3_6,77,81,403).  visible(fomdgt3_6,403). happensAt(running(fomdgt3_6),403).
orientation(fomdgt3_1,150,404). coords(fomdgt3_1,46,102,404).  visible(fomdgt3_1,404). happensAt(inactive(fomdgt3_1),404). orientation(fomdgt3_6,140,404). coords(fomdgt3_6,76,81,404).  visible(fomdgt3_6,404). happensAt(running(fomdgt3_6),404).
orientation(fomdgt3_1,150,405). coords(fomdgt3_1,46,102,405).  visible(fomdgt3_1,405). happensAt(inactive(fomdgt3_1),405). orientation(fomdgt3_6,140,405). coords(fomdgt3_6,75,81,405).  visible(fomdgt3_6,405). happensAt(running(fomdgt3_6),405).
orientation(fomdgt3_1,150,406). coords(fomdgt3_1,46,102,406).  visible(fomdgt3_1,406). happensAt(inactive(fomdgt3_1),406). orientation(fomdgt3_6,140,406). coords(fomdgt3_6,74,81,406).  visible(fomdgt3_6,406). happensAt(running(fomdgt3_6),406).
orientation(fomdgt3_1,150,407). coords(fomdgt3_1,46,102,407).  visible(fomdgt3_1,407). happensAt(inactive(fomdgt3_1),407). orientation(fomdgt3_6,140,407). coords(fomdgt3_6,71,82,407).  visible(fomdgt3_6,407). happensAt(running(fomdgt3_6),407).
orientation(fomdgt3_1,150,408). coords(fomdgt3_1,46,102,408).  visible(fomdgt3_1,408). happensAt(inactive(fomdgt3_1),408). orientation(fomdgt3_6,140,408). coords(fomdgt3_6,71,82,408).  visible(fomdgt3_6,408). happensAt(running(fomdgt3_6),408).
orientation(fomdgt3_1,150,409). coords(fomdgt3_1,46,102,409).  visible(fomdgt3_1,409). happensAt(inactive(fomdgt3_1),409). orientation(fomdgt3_6,140,409). coords(fomdgt3_6,69,82,409).  visible(fomdgt3_6,409). happensAt(running(fomdgt3_6),409).
orientation(fomdgt3_1,150,410). coords(fomdgt3_1,46,102,410).  visible(fomdgt3_1,410). happensAt(inactive(fomdgt3_1),410). orientation(fomdgt3_6,140,410). coords(fomdgt3_6,68,82,410).  visible(fomdgt3_6,410). happensAt(running(fomdgt3_6),410).
orientation(fomdgt3_1,150,411). coords(fomdgt3_1,46,102,411).  visible(fomdgt3_1,411). happensAt(inactive(fomdgt3_1),411). orientation(fomdgt3_6,140,411). coords(fomdgt3_6,68,82,411).  visible(fomdgt3_6,411). happensAt(running(fomdgt3_6),411).
orientation(fomdgt3_1,150,412). coords(fomdgt3_1,46,102,412).  visible(fomdgt3_1,412). happensAt(inactive(fomdgt3_1),412). orientation(fomdgt3_6,140,412). coords(fomdgt3_6,67,83,412).  visible(fomdgt3_6,412). happensAt(running(fomdgt3_6),412).
orientation(fomdgt3_1,150,413). coords(fomdgt3_1,46,102,413).  visible(fomdgt3_1,413). happensAt(inactive(fomdgt3_1),413). orientation(fomdgt3_6,140,413). coords(fomdgt3_6,66,83,413).  visible(fomdgt3_6,413). happensAt(running(fomdgt3_6),413).
orientation(fomdgt3_1,150,414). coords(fomdgt3_1,46,102,414).  visible(fomdgt3_1,414). happensAt(inactive(fomdgt3_1),414). orientation(fomdgt3_6,140,414). coords(fomdgt3_6,64,83,414).  visible(fomdgt3_6,414). happensAt(running(fomdgt3_6),414).
#show holdsAt/2.