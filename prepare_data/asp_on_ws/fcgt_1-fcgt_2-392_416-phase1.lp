
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

	orientation(fcgt_1,137,392). coords(fcgt_1,75,93,392).  visible(fcgt_1,392). happensAt(running(fcgt_1),392). orientation(fcgt_2,134,392). coords(fcgt_2,78,86,392).  visible(fcgt_2,392). happensAt(running(fcgt_2),392).
orientation(fcgt_1,137,393). coords(fcgt_1,75,91,393).  visible(fcgt_1,393). happensAt(running(fcgt_1),393). orientation(fcgt_2,130,393). coords(fcgt_2,78,85,393).  visible(fcgt_2,393). happensAt(running(fcgt_2),393).
orientation(fcgt_1,137,394). coords(fcgt_1,75,90,394).  visible(fcgt_1,394). happensAt(running(fcgt_1),394). orientation(fcgt_2,130,394). coords(fcgt_2,78,84,394).  visible(fcgt_2,394). happensAt(running(fcgt_2),394).
orientation(fcgt_1,137,395). coords(fcgt_1,75,90,395).  visible(fcgt_1,395). happensAt(running(fcgt_1),395). orientation(fcgt_2,130,395). coords(fcgt_2,78,82,395).  visible(fcgt_2,395). happensAt(running(fcgt_2),395).
orientation(fcgt_1,137,396). coords(fcgt_1,75,88,396).  visible(fcgt_1,396). happensAt(running(fcgt_1),396). orientation(fcgt_2,130,396). coords(fcgt_2,78,81,396).  visible(fcgt_2,396). happensAt(running(fcgt_2),396).
orientation(fcgt_1,137,397). coords(fcgt_1,73,87,397).  visible(fcgt_1,397). happensAt(running(fcgt_1),397). orientation(fcgt_2,130,397). coords(fcgt_2,78,81,397).  visible(fcgt_2,397). happensAt(running(fcgt_2),397).
orientation(fcgt_1,137,398). coords(fcgt_1,73,86,398).  visible(fcgt_1,398). happensAt(running(fcgt_1),398). orientation(fcgt_2,130,398). coords(fcgt_2,77,79,398).  visible(fcgt_2,398). happensAt(running(fcgt_2),398).
orientation(fcgt_1,137,399). coords(fcgt_1,72,85,399).  visible(fcgt_1,399). happensAt(running(fcgt_1),399). orientation(fcgt_2,130,399). coords(fcgt_2,77,78,399).  visible(fcgt_2,399). happensAt(running(fcgt_2),399).
orientation(fcgt_1,137,400). coords(fcgt_1,72,85,400).  visible(fcgt_1,400). happensAt(running(fcgt_1),400). orientation(fcgt_2,130,400). coords(fcgt_2,77,77,400).  visible(fcgt_2,400). happensAt(running(fcgt_2),400).
orientation(fcgt_1,132,401). coords(fcgt_1,72,85,401).  visible(fcgt_1,401). happensAt(running(fcgt_1),401). orientation(fcgt_2,130,401). coords(fcgt_2,77,75,401).  visible(fcgt_2,401). happensAt(running(fcgt_2),401).
orientation(fcgt_1,128,402). coords(fcgt_1,72,84,402).  visible(fcgt_1,402). happensAt(running(fcgt_1),402). orientation(fcgt_2,130,402). coords(fcgt_2,77,74,402).  visible(fcgt_2,402). happensAt(running(fcgt_2),402).
orientation(fcgt_1,128,403). coords(fcgt_1,72,82,403).  visible(fcgt_1,403). happensAt(running(fcgt_1),403). orientation(fcgt_2,130,403). coords(fcgt_2,77,74,403).  visible(fcgt_2,403). happensAt(running(fcgt_2),403).
orientation(fcgt_1,128,404). coords(fcgt_1,72,80,404).  visible(fcgt_1,404). happensAt(running(fcgt_1),404). orientation(fcgt_2,130,404). coords(fcgt_2,78,72,404).  visible(fcgt_2,404). happensAt(running(fcgt_2),404).
orientation(fcgt_1,128,405). coords(fcgt_1,72,78,405).  visible(fcgt_1,405). happensAt(running(fcgt_1),405). orientation(fcgt_2,130,405). coords(fcgt_2,78,71,405).  visible(fcgt_2,405). happensAt(running(fcgt_2),405).
orientation(fcgt_1,128,406). coords(fcgt_1,72,77,406).  visible(fcgt_1,406). happensAt(running(fcgt_1),406). orientation(fcgt_2,130,406). coords(fcgt_2,78,69,406).  visible(fcgt_2,406). happensAt(running(fcgt_2),406).
orientation(fcgt_1,128,407). coords(fcgt_1,72,77,407).  visible(fcgt_1,407). happensAt(running(fcgt_1),407). orientation(fcgt_2,130,407). coords(fcgt_2,78,68,407).  visible(fcgt_2,407). happensAt(running(fcgt_2),407).
orientation(fcgt_1,128,408). coords(fcgt_1,72,75,408).  visible(fcgt_1,408). happensAt(running(fcgt_1),408). orientation(fcgt_2,130,408). coords(fcgt_2,78,68,408).  visible(fcgt_2,408). happensAt(running(fcgt_2),408).
orientation(fcgt_1,128,409). coords(fcgt_1,72,75,409).  visible(fcgt_1,409). happensAt(running(fcgt_1),409). orientation(fcgt_2,130,409). coords(fcgt_2,79,68,409).  visible(fcgt_2,409). happensAt(running(fcgt_2),409).
orientation(fcgt_1,128,410). coords(fcgt_1,73,74,410).  visible(fcgt_1,410). happensAt(running(fcgt_1),410). orientation(fcgt_2,130,410). coords(fcgt_2,79,67,410).  visible(fcgt_2,410). happensAt(running(fcgt_2),410).
orientation(fcgt_1,128,411). coords(fcgt_1,73,73,411).  visible(fcgt_1,411). happensAt(running(fcgt_1),411). orientation(fcgt_2,130,411). coords(fcgt_2,78,67,411).  visible(fcgt_2,411). happensAt(running(fcgt_2),411).
orientation(fcgt_1,128,412). coords(fcgt_1,73,71,412).  visible(fcgt_1,412). happensAt(running(fcgt_1),412). orientation(fcgt_2,130,412). coords(fcgt_2,78,65,412).  visible(fcgt_2,412). happensAt(running(fcgt_2),412).
orientation(fcgt_1,128,413). coords(fcgt_1,73,70,413).  visible(fcgt_1,413). happensAt(running(fcgt_1),413). orientation(fcgt_2,130,413). coords(fcgt_2,78,64,413).  visible(fcgt_2,413). happensAt(running(fcgt_2),413).
orientation(fcgt_1,128,414). coords(fcgt_1,73,69,414).  visible(fcgt_1,414). happensAt(running(fcgt_1),414). orientation(fcgt_2,130,414). coords(fcgt_2,79,63,414).  visible(fcgt_2,414). happensAt(running(fcgt_2),414).
orientation(fcgt_1,128,415). coords(fcgt_1,73,68,415).  visible(fcgt_1,415). happensAt(running(fcgt_1),415). orientation(fcgt_2,130,415). coords(fcgt_2,81,64,415).  visible(fcgt_2,415). happensAt(running(fcgt_2),415).
orientation(fcgt_1,128,416). coords(fcgt_1,74,68,416).  visible(fcgt_1,416). happensAt(running(fcgt_1),416). orientation(fcgt_2,130,416). coords(fcgt_2,82,63,416).  happensAt(disappear(fcgt_2),416). happensAt(running(fcgt_2),416).
#show holdsAt/2.