
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

	orientation(wk1gt_4,156,387). coords(wk1gt_4,52,174,387).  visible(wk1gt_4,387). happensAt(active(wk1gt_4),387). orientation(wk1gt_5,162,387). coords(wk1gt_5,50,163,387).  visible(wk1gt_5,387). happensAt(inactive(wk1gt_5),387).
orientation(wk1gt_4,155,388). coords(wk1gt_4,52,174,388).  visible(wk1gt_4,388). happensAt(active(wk1gt_4),388). orientation(wk1gt_5,162,388). coords(wk1gt_5,50,163,388).  visible(wk1gt_5,388). happensAt(inactive(wk1gt_5),388).
orientation(wk1gt_4,156,389). coords(wk1gt_4,52,173,389).  visible(wk1gt_4,389). happensAt(active(wk1gt_4),389). orientation(wk1gt_5,162,389). coords(wk1gt_5,50,163,389).  visible(wk1gt_5,389). happensAt(inactive(wk1gt_5),389).
orientation(wk1gt_4,155,390). coords(wk1gt_4,52,173,390).  visible(wk1gt_4,390). happensAt(active(wk1gt_4),390). orientation(wk1gt_5,162,390). coords(wk1gt_5,51,163,390).  visible(wk1gt_5,390). happensAt(inactive(wk1gt_5),390).
orientation(wk1gt_4,162,391). coords(wk1gt_4,52,173,391).  visible(wk1gt_4,391). happensAt(active(wk1gt_4),391). orientation(wk1gt_5,162,391). coords(wk1gt_5,51,163,391).  visible(wk1gt_5,391). happensAt(inactive(wk1gt_5),391).
orientation(wk1gt_4,159,392). coords(wk1gt_4,52,173,392).  visible(wk1gt_4,392). happensAt(active(wk1gt_4),392). orientation(wk1gt_5,162,392). coords(wk1gt_5,50,163,392).  visible(wk1gt_5,392). happensAt(inactive(wk1gt_5),392).
orientation(wk1gt_4,152,393). coords(wk1gt_4,52,173,393).  visible(wk1gt_4,393). happensAt(active(wk1gt_4),393). orientation(wk1gt_5,162,393). coords(wk1gt_5,50,163,393).  visible(wk1gt_5,393). happensAt(inactive(wk1gt_5),393).
orientation(wk1gt_4,155,394). coords(wk1gt_4,52,174,394).  visible(wk1gt_4,394). happensAt(active(wk1gt_4),394). orientation(wk1gt_5,162,394). coords(wk1gt_5,50,163,394).  visible(wk1gt_5,394). happensAt(inactive(wk1gt_5),394).
orientation(wk1gt_4,158,395). coords(wk1gt_4,52,174,395).  visible(wk1gt_4,395). happensAt(active(wk1gt_4),395). orientation(wk1gt_5,162,395). coords(wk1gt_5,50,163,395).  visible(wk1gt_5,395). happensAt(inactive(wk1gt_5),395).
orientation(wk1gt_4,157,396). coords(wk1gt_4,52,174,396).  visible(wk1gt_4,396). happensAt(active(wk1gt_4),396). orientation(wk1gt_5,162,396). coords(wk1gt_5,50,163,396).  visible(wk1gt_5,396). happensAt(inactive(wk1gt_5),396).
orientation(wk1gt_4,159,397). coords(wk1gt_4,52,175,397).  visible(wk1gt_4,397). happensAt(active(wk1gt_4),397). orientation(wk1gt_5,162,397). coords(wk1gt_5,50,163,397).  visible(wk1gt_5,397). happensAt(inactive(wk1gt_5),397).
orientation(wk1gt_4,157,398). coords(wk1gt_4,52,175,398).  visible(wk1gt_4,398). happensAt(active(wk1gt_4),398). orientation(wk1gt_5,162,398). coords(wk1gt_5,50,163,398).  visible(wk1gt_5,398). happensAt(inactive(wk1gt_5),398).
orientation(wk1gt_4,163,399). coords(wk1gt_4,52,176,399).  visible(wk1gt_4,399). happensAt(active(wk1gt_4),399). orientation(wk1gt_5,162,399). coords(wk1gt_5,50,163,399).  visible(wk1gt_5,399). happensAt(inactive(wk1gt_5),399).
orientation(wk1gt_4,162,400). coords(wk1gt_4,52,176,400).  visible(wk1gt_4,400). happensAt(active(wk1gt_4),400). orientation(wk1gt_5,162,400). coords(wk1gt_5,50,163,400).  visible(wk1gt_5,400). happensAt(inactive(wk1gt_5),400).
orientation(wk1gt_4,168,401). coords(wk1gt_4,52,176,401).  visible(wk1gt_4,401). happensAt(active(wk1gt_4),401). orientation(wk1gt_5,162,401). coords(wk1gt_5,50,163,401).  visible(wk1gt_5,401). happensAt(inactive(wk1gt_5),401).
orientation(wk1gt_4,162,402). coords(wk1gt_4,52,176,402).  visible(wk1gt_4,402). happensAt(active(wk1gt_4),402). orientation(wk1gt_5,162,402). coords(wk1gt_5,50,163,402).  visible(wk1gt_5,402). happensAt(inactive(wk1gt_5),402).
orientation(wk1gt_4,162,403). coords(wk1gt_4,52,176,403).  visible(wk1gt_4,403). happensAt(active(wk1gt_4),403). orientation(wk1gt_5,162,403). coords(wk1gt_5,50,163,403).  visible(wk1gt_5,403). happensAt(inactive(wk1gt_5),403).
orientation(wk1gt_4,160,404). coords(wk1gt_4,52,176,404).  visible(wk1gt_4,404). happensAt(active(wk1gt_4),404). orientation(wk1gt_5,162,404). coords(wk1gt_5,50,163,404).  visible(wk1gt_5,404). happensAt(inactive(wk1gt_5),404).
orientation(wk1gt_4,154,405). coords(wk1gt_4,52,176,405).  visible(wk1gt_4,405). happensAt(active(wk1gt_4),405). orientation(wk1gt_5,162,405). coords(wk1gt_5,50,163,405).  visible(wk1gt_5,405). happensAt(inactive(wk1gt_5),405).
orientation(wk1gt_4,156,406). coords(wk1gt_4,52,176,406).  visible(wk1gt_4,406). happensAt(active(wk1gt_4),406). orientation(wk1gt_5,162,406). coords(wk1gt_5,50,163,406).  visible(wk1gt_5,406). happensAt(inactive(wk1gt_5),406).
orientation(wk1gt_4,159,407). coords(wk1gt_4,52,176,407).  visible(wk1gt_4,407). happensAt(active(wk1gt_4),407). orientation(wk1gt_5,162,407). coords(wk1gt_5,50,163,407).  visible(wk1gt_5,407). happensAt(inactive(wk1gt_5),407).
orientation(wk1gt_4,156,408). coords(wk1gt_4,52,176,408).  visible(wk1gt_4,408). happensAt(active(wk1gt_4),408). orientation(wk1gt_5,162,408). coords(wk1gt_5,50,163,408).  visible(wk1gt_5,408). happensAt(inactive(wk1gt_5),408).
orientation(wk1gt_4,155,409). coords(wk1gt_4,52,176,409).  visible(wk1gt_4,409). happensAt(active(wk1gt_4),409). orientation(wk1gt_5,162,409). coords(wk1gt_5,50,163,409).  visible(wk1gt_5,409). happensAt(inactive(wk1gt_5),409).
orientation(wk1gt_4,162,410). coords(wk1gt_4,52,176,410).  visible(wk1gt_4,410). happensAt(active(wk1gt_4),410). orientation(wk1gt_5,162,410). coords(wk1gt_5,50,163,410).  visible(wk1gt_5,410). happensAt(inactive(wk1gt_5),410).
orientation(wk1gt_4,168,411). coords(wk1gt_4,52,176,411).  visible(wk1gt_4,411). happensAt(active(wk1gt_4),411). orientation(wk1gt_5,162,411). coords(wk1gt_5,50,163,411).  visible(wk1gt_5,411). happensAt(inactive(wk1gt_5),411).
#show holdsAt/2.