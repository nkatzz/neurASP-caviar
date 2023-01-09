
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

	orientation(fra2gt_3,162,394). coords(fra2gt_3,65,138,394).  visible(fra2gt_3,394). happensAt(walking(fra2gt_3),394). orientation(fra2gt_4,148,394). coords(fra2gt_4,77,138,394).  visible(fra2gt_4,394). happensAt(walking(fra2gt_4),394).
orientation(fra2gt_3,162,395). coords(fra2gt_3,65,140,395).  visible(fra2gt_3,395). happensAt(walking(fra2gt_3),395). orientation(fra2gt_4,148,395). coords(fra2gt_4,77,140,395).  visible(fra2gt_4,395). happensAt(walking(fra2gt_4),395).
orientation(fra2gt_3,164,396). coords(fra2gt_3,65,140,396).  visible(fra2gt_3,396). happensAt(walking(fra2gt_3),396). orientation(fra2gt_4,148,396). coords(fra2gt_4,77,140,396).  visible(fra2gt_4,396). happensAt(walking(fra2gt_4),396).
orientation(fra2gt_3,164,397). coords(fra2gt_3,65,141,397).  visible(fra2gt_3,397). happensAt(walking(fra2gt_3),397). orientation(fra2gt_4,143,397). coords(fra2gt_4,76,140,397).  visible(fra2gt_4,397). happensAt(walking(fra2gt_4),397).
orientation(fra2gt_3,164,398). coords(fra2gt_3,65,141,398).  visible(fra2gt_3,398). happensAt(walking(fra2gt_3),398). orientation(fra2gt_4,142,398). coords(fra2gt_4,75,140,398).  visible(fra2gt_4,398). happensAt(walking(fra2gt_4),398).
orientation(fra2gt_3,164,399). coords(fra2gt_3,65,141,399).  visible(fra2gt_3,399). happensAt(walking(fra2gt_3),399). orientation(fra2gt_4,138,399). coords(fra2gt_4,75,140,399).  visible(fra2gt_4,399). happensAt(walking(fra2gt_4),399).
orientation(fra2gt_3,164,400). coords(fra2gt_3,65,142,400).  visible(fra2gt_3,400). happensAt(walking(fra2gt_3),400). orientation(fra2gt_4,138,400). coords(fra2gt_4,75,140,400).  visible(fra2gt_4,400). happensAt(walking(fra2gt_4),400).
orientation(fra2gt_3,164,401). coords(fra2gt_3,65,143,401).  visible(fra2gt_3,401). happensAt(walking(fra2gt_3),401). orientation(fra2gt_4,138,401). coords(fra2gt_4,75,139,401).  visible(fra2gt_4,401). happensAt(walking(fra2gt_4),401).
orientation(fra2gt_3,164,402). coords(fra2gt_3,64,143,402).  visible(fra2gt_3,402). happensAt(walking(fra2gt_3),402). orientation(fra2gt_4,138,402). coords(fra2gt_4,75,139,402).  visible(fra2gt_4,402). happensAt(walking(fra2gt_4),402).
orientation(fra2gt_3,164,403). coords(fra2gt_3,64,145,403).  visible(fra2gt_3,403). happensAt(walking(fra2gt_3),403). orientation(fra2gt_4,138,403). coords(fra2gt_4,75,137,403).  visible(fra2gt_4,403). happensAt(walking(fra2gt_4),403).
orientation(fra2gt_3,164,404). coords(fra2gt_3,62,145,404).  visible(fra2gt_3,404). happensAt(walking(fra2gt_3),404). orientation(fra2gt_4,138,404). coords(fra2gt_4,75,134,404).  visible(fra2gt_4,404). happensAt(running(fra2gt_4),404).
orientation(fra2gt_3,164,405). coords(fra2gt_3,62,145,405).  visible(fra2gt_3,405). happensAt(walking(fra2gt_3),405). orientation(fra2gt_4,138,405). coords(fra2gt_4,75,133,405).  visible(fra2gt_4,405). happensAt(running(fra2gt_4),405).
orientation(fra2gt_3,164,406). coords(fra2gt_3,62,145,406).  visible(fra2gt_3,406). happensAt(walking(fra2gt_3),406). orientation(fra2gt_4,138,406). coords(fra2gt_4,75,131,406).  visible(fra2gt_4,406). happensAt(running(fra2gt_4),406).
orientation(fra2gt_3,164,407). coords(fra2gt_3,62,145,407).  visible(fra2gt_3,407). happensAt(walking(fra2gt_3),407). orientation(fra2gt_4,138,407). coords(fra2gt_4,75,131,407).  visible(fra2gt_4,407). happensAt(running(fra2gt_4),407).
orientation(fra2gt_3,164,408). coords(fra2gt_3,62,145,408).  visible(fra2gt_3,408). happensAt(walking(fra2gt_3),408). orientation(fra2gt_4,147,408). coords(fra2gt_4,75,129,408).  visible(fra2gt_4,408). happensAt(running(fra2gt_4),408).
orientation(fra2gt_3,164,409). coords(fra2gt_3,61,146,409).  visible(fra2gt_3,409). happensAt(walking(fra2gt_3),409). orientation(fra2gt_4,147,409). coords(fra2gt_4,75,127,409).  visible(fra2gt_4,409). happensAt(running(fra2gt_4),409).
orientation(fra2gt_3,164,410). coords(fra2gt_3,60,146,410).  visible(fra2gt_3,410). happensAt(walking(fra2gt_3),410). orientation(fra2gt_4,147,410). coords(fra2gt_4,75,125,410).  visible(fra2gt_4,410). happensAt(running(fra2gt_4),410).
orientation(fra2gt_3,164,411). coords(fra2gt_3,59,148,411).  visible(fra2gt_3,411). happensAt(walking(fra2gt_3),411). orientation(fra2gt_4,147,411). coords(fra2gt_4,74,125,411).  visible(fra2gt_4,411). happensAt(running(fra2gt_4),411).
orientation(fra2gt_3,164,412). coords(fra2gt_3,59,149,412).  visible(fra2gt_3,412). happensAt(walking(fra2gt_3),412). orientation(fra2gt_4,147,412). coords(fra2gt_4,74,124,412).  visible(fra2gt_4,412). happensAt(running(fra2gt_4),412).
orientation(fra2gt_3,164,413). coords(fra2gt_3,58,150,413).  visible(fra2gt_3,413). happensAt(walking(fra2gt_3),413). orientation(fra2gt_4,147,413). coords(fra2gt_4,74,123,413).  visible(fra2gt_4,413). happensAt(running(fra2gt_4),413).
orientation(fra2gt_3,164,414). coords(fra2gt_3,58,152,414).  visible(fra2gt_3,414). happensAt(walking(fra2gt_3),414). orientation(fra2gt_4,147,414). coords(fra2gt_4,74,123,414).  visible(fra2gt_4,414). happensAt(running(fra2gt_4),414).
orientation(fra2gt_3,164,415). coords(fra2gt_3,58,153,415).  visible(fra2gt_3,415). happensAt(walking(fra2gt_3),415). orientation(fra2gt_4,137,415). coords(fra2gt_4,74,122,415).  visible(fra2gt_4,415). happensAt(running(fra2gt_4),415).
orientation(fra2gt_3,164,416). coords(fra2gt_3,57,153,416).  visible(fra2gt_3,416). happensAt(walking(fra2gt_3),416). orientation(fra2gt_4,137,416). coords(fra2gt_4,74,120,416).  visible(fra2gt_4,416). happensAt(running(fra2gt_4),416).
orientation(fra2gt_3,164,417). coords(fra2gt_3,57,153,417).  visible(fra2gt_3,417). happensAt(walking(fra2gt_3),417). orientation(fra2gt_4,137,417). coords(fra2gt_4,74,119,417).  visible(fra2gt_4,417). happensAt(running(fra2gt_4),417).
orientation(fra2gt_3,164,418). coords(fra2gt_3,57,154,418).  visible(fra2gt_3,418). happensAt(walking(fra2gt_3),418). orientation(fra2gt_4,137,418). coords(fra2gt_4,74,118,418).  visible(fra2gt_4,418). happensAt(running(fra2gt_4),418).
#show holdsAt/2.