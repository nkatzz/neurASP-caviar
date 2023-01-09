
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

	orientation(fcgt_1,146,388). coords(fcgt_1,77,96,388).  visible(fcgt_1,388). happensAt(running(fcgt_1),388). orientation(fcgt_4,166,388). coords(fcgt_4,82,206,388).  visible(fcgt_4,388). happensAt(inactive(fcgt_4),388).
orientation(fcgt_1,140,389). coords(fcgt_1,76,95,389).  visible(fcgt_1,389). happensAt(running(fcgt_1),389). orientation(fcgt_4,166,389). coords(fcgt_4,82,206,389).  visible(fcgt_4,389). happensAt(inactive(fcgt_4),389).
orientation(fcgt_1,137,390). coords(fcgt_1,76,95,390).  visible(fcgt_1,390). happensAt(running(fcgt_1),390). orientation(fcgt_4,166,390). coords(fcgt_4,82,206,390).  visible(fcgt_4,390). happensAt(inactive(fcgt_4),390).
orientation(fcgt_1,137,391). coords(fcgt_1,75,93,391).  visible(fcgt_1,391). happensAt(running(fcgt_1),391). orientation(fcgt_4,166,391). coords(fcgt_4,82,206,391).  visible(fcgt_4,391). happensAt(inactive(fcgt_4),391).
orientation(fcgt_1,137,392). coords(fcgt_1,75,93,392).  visible(fcgt_1,392). happensAt(running(fcgt_1),392). orientation(fcgt_4,166,392). coords(fcgt_4,82,206,392).  visible(fcgt_4,392). happensAt(inactive(fcgt_4),392).
orientation(fcgt_1,137,393). coords(fcgt_1,75,91,393).  visible(fcgt_1,393). happensAt(running(fcgt_1),393). orientation(fcgt_4,166,393). coords(fcgt_4,82,206,393).  visible(fcgt_4,393). happensAt(inactive(fcgt_4),393).
orientation(fcgt_1,137,394). coords(fcgt_1,75,90,394).  visible(fcgt_1,394). happensAt(running(fcgt_1),394). orientation(fcgt_4,166,394). coords(fcgt_4,82,206,394).  visible(fcgt_4,394). happensAt(inactive(fcgt_4),394).
orientation(fcgt_1,137,395). coords(fcgt_1,75,90,395).  visible(fcgt_1,395). happensAt(running(fcgt_1),395). orientation(fcgt_4,166,395). coords(fcgt_4,82,206,395).  visible(fcgt_4,395). happensAt(inactive(fcgt_4),395).
orientation(fcgt_1,137,396). coords(fcgt_1,75,88,396).  visible(fcgt_1,396). happensAt(running(fcgt_1),396). orientation(fcgt_4,166,396). coords(fcgt_4,82,206,396).  visible(fcgt_4,396). happensAt(inactive(fcgt_4),396).
orientation(fcgt_1,137,397). coords(fcgt_1,73,87,397).  visible(fcgt_1,397). happensAt(running(fcgt_1),397). orientation(fcgt_4,166,397). coords(fcgt_4,82,206,397).  visible(fcgt_4,397). happensAt(inactive(fcgt_4),397).
orientation(fcgt_1,137,398). coords(fcgt_1,73,86,398).  visible(fcgt_1,398). happensAt(running(fcgt_1),398). orientation(fcgt_4,166,398). coords(fcgt_4,82,206,398).  visible(fcgt_4,398). happensAt(inactive(fcgt_4),398).
orientation(fcgt_1,137,399). coords(fcgt_1,72,85,399).  visible(fcgt_1,399). happensAt(running(fcgt_1),399). orientation(fcgt_4,166,399). coords(fcgt_4,82,206,399).  visible(fcgt_4,399). happensAt(inactive(fcgt_4),399).
orientation(fcgt_1,137,400). coords(fcgt_1,72,85,400).  visible(fcgt_1,400). happensAt(running(fcgt_1),400). orientation(fcgt_4,166,400). coords(fcgt_4,82,206,400).  visible(fcgt_4,400). happensAt(inactive(fcgt_4),400).
orientation(fcgt_1,132,401). coords(fcgt_1,72,85,401).  visible(fcgt_1,401). happensAt(running(fcgt_1),401). orientation(fcgt_4,166,401). coords(fcgt_4,82,206,401).  visible(fcgt_4,401). happensAt(inactive(fcgt_4),401).
orientation(fcgt_1,128,402). coords(fcgt_1,72,84,402).  visible(fcgt_1,402). happensAt(running(fcgt_1),402). orientation(fcgt_4,166,402). coords(fcgt_4,82,206,402).  visible(fcgt_4,402). happensAt(inactive(fcgt_4),402).
orientation(fcgt_1,128,403). coords(fcgt_1,72,82,403).  visible(fcgt_1,403). happensAt(running(fcgt_1),403). orientation(fcgt_4,166,403). coords(fcgt_4,82,206,403).  visible(fcgt_4,403). happensAt(inactive(fcgt_4),403).
orientation(fcgt_1,128,404). coords(fcgt_1,72,80,404).  visible(fcgt_1,404). happensAt(running(fcgt_1),404). orientation(fcgt_4,166,404). coords(fcgt_4,82,206,404).  visible(fcgt_4,404). happensAt(inactive(fcgt_4),404).
orientation(fcgt_1,128,405). coords(fcgt_1,72,78,405).  visible(fcgt_1,405). happensAt(running(fcgt_1),405). orientation(fcgt_4,166,405). coords(fcgt_4,82,206,405).  visible(fcgt_4,405). happensAt(inactive(fcgt_4),405).
orientation(fcgt_1,128,406). coords(fcgt_1,72,77,406).  visible(fcgt_1,406). happensAt(running(fcgt_1),406). orientation(fcgt_4,166,406). coords(fcgt_4,82,206,406).  visible(fcgt_4,406). happensAt(inactive(fcgt_4),406).
orientation(fcgt_1,128,407). coords(fcgt_1,72,77,407).  visible(fcgt_1,407). happensAt(running(fcgt_1),407). orientation(fcgt_4,166,407). coords(fcgt_4,82,206,407).  visible(fcgt_4,407). happensAt(inactive(fcgt_4),407).
orientation(fcgt_1,128,408). coords(fcgt_1,72,75,408).  visible(fcgt_1,408). happensAt(running(fcgt_1),408). orientation(fcgt_4,166,408). coords(fcgt_4,82,206,408).  visible(fcgt_4,408). happensAt(inactive(fcgt_4),408).
orientation(fcgt_1,128,409). coords(fcgt_1,72,75,409).  visible(fcgt_1,409). happensAt(running(fcgt_1),409). orientation(fcgt_4,166,409). coords(fcgt_4,82,206,409).  visible(fcgt_4,409). happensAt(inactive(fcgt_4),409).
orientation(fcgt_1,128,410). coords(fcgt_1,73,74,410).  visible(fcgt_1,410). happensAt(running(fcgt_1),410). orientation(fcgt_4,166,410). coords(fcgt_4,82,206,410).  visible(fcgt_4,410). happensAt(inactive(fcgt_4),410).
orientation(fcgt_1,128,411). coords(fcgt_1,73,73,411).  visible(fcgt_1,411). happensAt(running(fcgt_1),411). orientation(fcgt_4,166,411). coords(fcgt_4,82,206,411).  visible(fcgt_4,411). happensAt(inactive(fcgt_4),411).
orientation(fcgt_1,128,412). coords(fcgt_1,73,71,412).  visible(fcgt_1,412). happensAt(running(fcgt_1),412). orientation(fcgt_4,166,412). coords(fcgt_4,82,206,412).  visible(fcgt_4,412). happensAt(inactive(fcgt_4),412).
#show holdsAt/2.