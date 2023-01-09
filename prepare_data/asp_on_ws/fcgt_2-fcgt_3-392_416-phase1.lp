
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
orientation(fcgt_2,130,403). coords(fcgt_2,77,74,403).  visible(fcgt_2,403). happensAt(running(fcgt_2),403). orientation(fcgt_3,164,403). coords(fcgt_3,99,198,403).  visible(fcgt_3,403). happensAt(walking(fcgt_3),403).
orientation(fcgt_2,130,404). coords(fcgt_2,78,72,404).  visible(fcgt_2,404). happensAt(running(fcgt_2),404). orientation(fcgt_3,164,404). coords(fcgt_3,99,198,404).  visible(fcgt_3,404). happensAt(walking(fcgt_3),404).
orientation(fcgt_2,130,405). coords(fcgt_2,78,71,405).  visible(fcgt_2,405). happensAt(running(fcgt_2),405). orientation(fcgt_3,164,405). coords(fcgt_3,99,198,405).  visible(fcgt_3,405). happensAt(walking(fcgt_3),405).
orientation(fcgt_2,130,406). coords(fcgt_2,78,69,406).  visible(fcgt_2,406). happensAt(running(fcgt_2),406). orientation(fcgt_3,164,406). coords(fcgt_3,99,199,406).  visible(fcgt_3,406). happensAt(walking(fcgt_3),406).
orientation(fcgt_2,130,407). coords(fcgt_2,78,68,407).  visible(fcgt_2,407). happensAt(running(fcgt_2),407). orientation(fcgt_3,164,407). coords(fcgt_3,99,199,407).  visible(fcgt_3,407). happensAt(walking(fcgt_3),407).
orientation(fcgt_2,130,408). coords(fcgt_2,78,68,408).  visible(fcgt_2,408). happensAt(running(fcgt_2),408). orientation(fcgt_3,164,408). coords(fcgt_3,99,200,408).  visible(fcgt_3,408). happensAt(walking(fcgt_3),408).
orientation(fcgt_2,130,409). coords(fcgt_2,79,68,409).  visible(fcgt_2,409). happensAt(running(fcgt_2),409). orientation(fcgt_3,164,409). coords(fcgt_3,99,200,409).  visible(fcgt_3,409). happensAt(walking(fcgt_3),409).
orientation(fcgt_2,130,410). coords(fcgt_2,79,67,410).  visible(fcgt_2,410). happensAt(running(fcgt_2),410). orientation(fcgt_3,164,410). coords(fcgt_3,99,200,410).  visible(fcgt_3,410). happensAt(walking(fcgt_3),410).
orientation(fcgt_2,130,411). coords(fcgt_2,78,67,411).  visible(fcgt_2,411). happensAt(running(fcgt_2),411). orientation(fcgt_3,164,411). coords(fcgt_3,99,200,411).  visible(fcgt_3,411). happensAt(walking(fcgt_3),411).
orientation(fcgt_2,130,412). coords(fcgt_2,78,65,412).  visible(fcgt_2,412). happensAt(running(fcgt_2),412). orientation(fcgt_3,164,412). coords(fcgt_3,98,200,412).  visible(fcgt_3,412). happensAt(walking(fcgt_3),412).
orientation(fcgt_2,130,413). coords(fcgt_2,78,64,413).  visible(fcgt_2,413). happensAt(running(fcgt_2),413). orientation(fcgt_3,164,413). coords(fcgt_3,98,200,413).  visible(fcgt_3,413). happensAt(walking(fcgt_3),413).
orientation(fcgt_2,130,414). coords(fcgt_2,79,63,414).  visible(fcgt_2,414). happensAt(running(fcgt_2),414). orientation(fcgt_3,164,414). coords(fcgt_3,98,200,414).  visible(fcgt_3,414). happensAt(walking(fcgt_3),414).
orientation(fcgt_2,130,415). coords(fcgt_2,81,64,415).  visible(fcgt_2,415). happensAt(running(fcgt_2),415). orientation(fcgt_3,164,415). coords(fcgt_3,98,201,415).  visible(fcgt_3,415). happensAt(walking(fcgt_3),415).
orientation(fcgt_2,130,416). coords(fcgt_2,82,63,416).  happensAt(disappear(fcgt_2),416). happensAt(running(fcgt_2),416). orientation(fcgt_3,164,416). coords(fcgt_3,98,201,416).  visible(fcgt_3,416). happensAt(walking(fcgt_3),416).
#show holdsAt/2.