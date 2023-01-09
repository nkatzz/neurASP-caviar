
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

	orientation(br1gt_2,7,394). coords(br1gt_2,86,208,394).  visible(br1gt_2,394). happensAt(inactive(br1gt_2),394). orientation(br1gt_3,162,394). coords(br1gt_3,41,157,394).  visible(br1gt_3,394). happensAt(walking(br1gt_3),394).
orientation(br1gt_2,7,395). coords(br1gt_2,86,207,395).  visible(br1gt_2,395). happensAt(inactive(br1gt_2),395). orientation(br1gt_3,162,395). coords(br1gt_3,41,156,395).  visible(br1gt_3,395). happensAt(walking(br1gt_3),395).
orientation(br1gt_2,7,396). coords(br1gt_2,86,207,396).  visible(br1gt_2,396). happensAt(inactive(br1gt_2),396). orientation(br1gt_3,162,396). coords(br1gt_3,41,157,396).  visible(br1gt_3,396). happensAt(walking(br1gt_3),396).
orientation(br1gt_2,7,397). coords(br1gt_2,84,207,397).  visible(br1gt_2,397). happensAt(inactive(br1gt_2),397). orientation(br1gt_3,162,397). coords(br1gt_3,41,157,397).  visible(br1gt_3,397). happensAt(walking(br1gt_3),397).
orientation(br1gt_2,7,398). coords(br1gt_2,84,207,398).  visible(br1gt_2,398). happensAt(inactive(br1gt_2),398). orientation(br1gt_3,162,398). coords(br1gt_3,42,156,398).  visible(br1gt_3,398). happensAt(walking(br1gt_3),398).
orientation(br1gt_2,7,399). coords(br1gt_2,84,207,399).  visible(br1gt_2,399). happensAt(inactive(br1gt_2),399). orientation(br1gt_3,162,399). coords(br1gt_3,43,156,399).  visible(br1gt_3,399). happensAt(walking(br1gt_3),399).
orientation(br1gt_2,7,400). coords(br1gt_2,84,207,400).  visible(br1gt_2,400). happensAt(inactive(br1gt_2),400). orientation(br1gt_3,162,400). coords(br1gt_3,43,156,400).  visible(br1gt_3,400). happensAt(walking(br1gt_3),400).
orientation(br1gt_2,7,401). coords(br1gt_2,84,207,401).  visible(br1gt_2,401). happensAt(inactive(br1gt_2),401). orientation(br1gt_3,162,401). coords(br1gt_3,43,156,401).  visible(br1gt_3,401). happensAt(walking(br1gt_3),401).
orientation(br1gt_2,7,402). coords(br1gt_2,84,208,402).  visible(br1gt_2,402). happensAt(inactive(br1gt_2),402). orientation(br1gt_3,162,402). coords(br1gt_3,43,156,402).  visible(br1gt_3,402). happensAt(walking(br1gt_3),402).
orientation(br1gt_2,7,403). coords(br1gt_2,84,208,403).  visible(br1gt_2,403). happensAt(inactive(br1gt_2),403). orientation(br1gt_3,162,403). coords(br1gt_3,44,156,403).  visible(br1gt_3,403). happensAt(walking(br1gt_3),403).
orientation(br1gt_2,7,404). coords(br1gt_2,84,208,404).  visible(br1gt_2,404). happensAt(inactive(br1gt_2),404). orientation(br1gt_3,162,404). coords(br1gt_3,44,156,404).  visible(br1gt_3,404). happensAt(walking(br1gt_3),404).
orientation(br1gt_2,7,405). coords(br1gt_2,84,208,405).  visible(br1gt_2,405). happensAt(inactive(br1gt_2),405). orientation(br1gt_3,162,405). coords(br1gt_3,44,156,405).  visible(br1gt_3,405). happensAt(walking(br1gt_3),405).
orientation(br1gt_2,7,406). coords(br1gt_2,84,208,406).  visible(br1gt_2,406). happensAt(inactive(br1gt_2),406). orientation(br1gt_3,162,406). coords(br1gt_3,44,156,406).  visible(br1gt_3,406). happensAt(walking(br1gt_3),406).
orientation(br1gt_2,7,407). coords(br1gt_2,84,208,407).  visible(br1gt_2,407). happensAt(inactive(br1gt_2),407). orientation(br1gt_3,162,407). coords(br1gt_3,45,156,407).  visible(br1gt_3,407). happensAt(walking(br1gt_3),407).
orientation(br1gt_2,7,408). coords(br1gt_2,84,208,408).  visible(br1gt_2,408). happensAt(inactive(br1gt_2),408). orientation(br1gt_3,162,408). coords(br1gt_3,45,156,408).  visible(br1gt_3,408). happensAt(walking(br1gt_3),408).
orientation(br1gt_2,7,409). coords(br1gt_2,84,208,409).  visible(br1gt_2,409). happensAt(inactive(br1gt_2),409). orientation(br1gt_3,162,409). coords(br1gt_3,46,156,409).  visible(br1gt_3,409). happensAt(walking(br1gt_3),409).
orientation(br1gt_2,7,410). coords(br1gt_2,84,208,410).  visible(br1gt_2,410). happensAt(inactive(br1gt_2),410). orientation(br1gt_3,162,410). coords(br1gt_3,46,156,410).  visible(br1gt_3,410). happensAt(walking(br1gt_3),410).
orientation(br1gt_2,10,411). coords(br1gt_2,84,208,411).  visible(br1gt_2,411). happensAt(inactive(br1gt_2),411). orientation(br1gt_3,162,411). coords(br1gt_3,47,156,411).  visible(br1gt_3,411). happensAt(walking(br1gt_3),411).
orientation(br1gt_2,10,412). coords(br1gt_2,84,208,412).  visible(br1gt_2,412). happensAt(inactive(br1gt_2),412). orientation(br1gt_3,162,412). coords(br1gt_3,47,156,412).  visible(br1gt_3,412). happensAt(walking(br1gt_3),412).
orientation(br1gt_2,10,413). coords(br1gt_2,84,208,413).  visible(br1gt_2,413). happensAt(inactive(br1gt_2),413). orientation(br1gt_3,162,413). coords(br1gt_3,49,156,413).  visible(br1gt_3,413). happensAt(walking(br1gt_3),413).
orientation(br1gt_2,10,414). coords(br1gt_2,84,208,414).  visible(br1gt_2,414). happensAt(inactive(br1gt_2),414). orientation(br1gt_3,162,414). coords(br1gt_3,50,156,414).  visible(br1gt_3,414). happensAt(walking(br1gt_3),414).
orientation(br1gt_2,10,415). coords(br1gt_2,85,208,415).  visible(br1gt_2,415). happensAt(inactive(br1gt_2),415). orientation(br1gt_3,162,415). coords(br1gt_3,50,156,415).  visible(br1gt_3,415). happensAt(walking(br1gt_3),415).
orientation(br1gt_2,10,416). coords(br1gt_2,85,208,416).  visible(br1gt_2,416). happensAt(inactive(br1gt_2),416). orientation(br1gt_3,162,416). coords(br1gt_3,50,156,416).  visible(br1gt_3,416). happensAt(walking(br1gt_3),416).
orientation(br1gt_2,10,417). coords(br1gt_2,85,208,417).  visible(br1gt_2,417). happensAt(inactive(br1gt_2),417). orientation(br1gt_3,162,417). coords(br1gt_3,50,156,417).  visible(br1gt_3,417). happensAt(walking(br1gt_3),417).
orientation(br1gt_2,10,418). coords(br1gt_2,85,208,418).  visible(br1gt_2,418). happensAt(inactive(br1gt_2),418). orientation(br1gt_3,162,418). coords(br1gt_3,50,155,418).  visible(br1gt_3,418). happensAt(walking(br1gt_3),418).
#show holdsAt/2.