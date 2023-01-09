
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

	orientation(mws1gt_3,165,387). coords(mws1gt_3,39,161,387).  visible(mws1gt_3,387). happensAt(walking(mws1gt_3),387). orientation(mws1gt_7,174,387). coords(mws1gt_7,58,185,387).  visible(mws1gt_7,387). happensAt(walking(mws1gt_7),387).
orientation(mws1gt_3,165,388). coords(mws1gt_3,39,161,388).  visible(mws1gt_3,388). happensAt(walking(mws1gt_3),388). orientation(mws1gt_7,174,388). coords(mws1gt_7,59,185,388).  visible(mws1gt_7,388). happensAt(walking(mws1gt_7),388).
orientation(mws1gt_3,165,389). coords(mws1gt_3,39,161,389).  visible(mws1gt_3,389). happensAt(walking(mws1gt_3),389). orientation(mws1gt_7,174,389). coords(mws1gt_7,60,186,389).  visible(mws1gt_7,389). happensAt(walking(mws1gt_7),389).
orientation(mws1gt_3,165,390). coords(mws1gt_3,38,161,390).  visible(mws1gt_3,390). happensAt(walking(mws1gt_3),390). orientation(mws1gt_7,174,390). coords(mws1gt_7,60,186,390).  visible(mws1gt_7,390). happensAt(walking(mws1gt_7),390).
orientation(mws1gt_3,165,391). coords(mws1gt_3,38,161,391).  visible(mws1gt_3,391). happensAt(walking(mws1gt_3),391). orientation(mws1gt_7,174,391). coords(mws1gt_7,61,187,391).  visible(mws1gt_7,391). happensAt(walking(mws1gt_7),391).
orientation(mws1gt_3,165,392). coords(mws1gt_3,38,161,392).  visible(mws1gt_3,392). happensAt(walking(mws1gt_3),392). orientation(mws1gt_7,174,392). coords(mws1gt_7,61,187,392).  visible(mws1gt_7,392). happensAt(walking(mws1gt_7),392).
orientation(mws1gt_3,165,393). coords(mws1gt_3,38,161,393).  visible(mws1gt_3,393). happensAt(walking(mws1gt_3),393). orientation(mws1gt_7,174,393). coords(mws1gt_7,61,187,393).  visible(mws1gt_7,393). happensAt(walking(mws1gt_7),393).
orientation(mws1gt_3,165,394). coords(mws1gt_3,38,161,394).  visible(mws1gt_3,394). happensAt(walking(mws1gt_3),394). orientation(mws1gt_7,174,394). coords(mws1gt_7,61,187,394).  visible(mws1gt_7,394). happensAt(walking(mws1gt_7),394).
orientation(mws1gt_3,165,395). coords(mws1gt_3,37,161,395).  visible(mws1gt_3,395). happensAt(walking(mws1gt_3),395). orientation(mws1gt_7,174,395). coords(mws1gt_7,61,187,395).  visible(mws1gt_7,395). happensAt(walking(mws1gt_7),395).
orientation(mws1gt_3,165,396). coords(mws1gt_3,37,161,396).  visible(mws1gt_3,396). happensAt(walking(mws1gt_3),396). orientation(mws1gt_7,174,396). coords(mws1gt_7,61,187,396).  visible(mws1gt_7,396). happensAt(walking(mws1gt_7),396).
orientation(mws1gt_3,165,397). coords(mws1gt_3,37,161,397).  visible(mws1gt_3,397). happensAt(walking(mws1gt_3),397). orientation(mws1gt_7,174,397). coords(mws1gt_7,61,187,397).  visible(mws1gt_7,397). happensAt(walking(mws1gt_7),397).
orientation(mws1gt_3,165,398). coords(mws1gt_3,37,161,398).  visible(mws1gt_3,398). happensAt(walking(mws1gt_3),398). orientation(mws1gt_7,174,398). coords(mws1gt_7,61,187,398).  visible(mws1gt_7,398). happensAt(walking(mws1gt_7),398).
orientation(mws1gt_3,165,399). coords(mws1gt_3,37,161,399).  visible(mws1gt_3,399). happensAt(walking(mws1gt_3),399). orientation(mws1gt_7,174,399). coords(mws1gt_7,62,188,399).  visible(mws1gt_7,399). happensAt(walking(mws1gt_7),399).
orientation(mws1gt_3,165,400). coords(mws1gt_3,37,161,400).  visible(mws1gt_3,400). happensAt(walking(mws1gt_3),400). orientation(mws1gt_7,174,400). coords(mws1gt_7,63,188,400).  visible(mws1gt_7,400). happensAt(walking(mws1gt_7),400).
orientation(mws1gt_3,165,401). coords(mws1gt_3,36,161,401).  visible(mws1gt_3,401). happensAt(walking(mws1gt_3),401). orientation(mws1gt_7,174,401). coords(mws1gt_7,63,188,401).  visible(mws1gt_7,401). happensAt(walking(mws1gt_7),401).
orientation(mws1gt_3,165,402). coords(mws1gt_3,36,161,402).  visible(mws1gt_3,402). happensAt(walking(mws1gt_3),402). orientation(mws1gt_7,174,402). coords(mws1gt_7,65,188,402).  visible(mws1gt_7,402). happensAt(walking(mws1gt_7),402).
orientation(mws1gt_3,165,403). coords(mws1gt_3,36,161,403).  visible(mws1gt_3,403). happensAt(walking(mws1gt_3),403). orientation(mws1gt_7,174,403). coords(mws1gt_7,66,189,403).  visible(mws1gt_7,403). happensAt(walking(mws1gt_7),403).
orientation(mws1gt_3,165,404). coords(mws1gt_3,36,161,404).  visible(mws1gt_3,404). happensAt(walking(mws1gt_3),404). orientation(mws1gt_7,174,404). coords(mws1gt_7,66,189,404).  visible(mws1gt_7,404). happensAt(walking(mws1gt_7),404).
orientation(mws1gt_3,165,405). coords(mws1gt_3,36,161,405).  visible(mws1gt_3,405). happensAt(walking(mws1gt_3),405). orientation(mws1gt_7,174,405). coords(mws1gt_7,66,189,405).  visible(mws1gt_7,405). happensAt(walking(mws1gt_7),405).
orientation(mws1gt_3,165,406). coords(mws1gt_3,36,161,406).  visible(mws1gt_3,406). happensAt(walking(mws1gt_3),406). orientation(mws1gt_7,174,406). coords(mws1gt_7,66,189,406).  visible(mws1gt_7,406). happensAt(walking(mws1gt_7),406).
orientation(mws1gt_3,165,407). coords(mws1gt_3,35,161,407).  visible(mws1gt_3,407). happensAt(walking(mws1gt_3),407). orientation(mws1gt_7,174,407). coords(mws1gt_7,67,190,407).  visible(mws1gt_7,407). happensAt(walking(mws1gt_7),407).
orientation(mws1gt_3,165,408). coords(mws1gt_3,35,161,408).  visible(mws1gt_3,408). happensAt(walking(mws1gt_3),408). orientation(mws1gt_7,174,408). coords(mws1gt_7,67,190,408).  visible(mws1gt_7,408). happensAt(walking(mws1gt_7),408).
orientation(mws1gt_3,165,409). coords(mws1gt_3,34,160,409).  visible(mws1gt_3,409). happensAt(walking(mws1gt_3),409). orientation(mws1gt_7,174,409). coords(mws1gt_7,67,190,409).  visible(mws1gt_7,409). happensAt(walking(mws1gt_7),409).
orientation(mws1gt_3,165,410). coords(mws1gt_3,34,160,410).  visible(mws1gt_3,410). happensAt(walking(mws1gt_3),410). orientation(mws1gt_7,174,410). coords(mws1gt_7,67,190,410).  visible(mws1gt_7,410). happensAt(walking(mws1gt_7),410).
orientation(mws1gt_3,165,411). coords(mws1gt_3,34,160,411).  visible(mws1gt_3,411). happensAt(walking(mws1gt_3),411). orientation(mws1gt_7,174,411). coords(mws1gt_7,67,190,411).  visible(mws1gt_7,411). happensAt(walking(mws1gt_7),411).
#show holdsAt/2.