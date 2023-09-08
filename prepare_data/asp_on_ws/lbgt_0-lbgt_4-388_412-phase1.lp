
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

	orientation(lbgt_0,34,388). coords(lbgt_0,269,252,388).  visible(lbgt_0,388). happensAt(walking(lbgt_0),388). orientation(lbgt_4,147,388). coords(lbgt_4,80,68,388).  visible(lbgt_4,388). happensAt(walking(lbgt_4),388).
orientation(lbgt_0,34,389). coords(lbgt_0,270,254,389).  visible(lbgt_0,389). happensAt(walking(lbgt_0),389). orientation(lbgt_4,147,389). coords(lbgt_4,80,69,389).  visible(lbgt_4,389). happensAt(walking(lbgt_4),389).
orientation(lbgt_0,34,390). coords(lbgt_0,271,256,390).  visible(lbgt_0,390). happensAt(walking(lbgt_0),390). orientation(lbgt_4,147,390). coords(lbgt_4,80,70,390).  visible(lbgt_4,390). happensAt(walking(lbgt_4),390).
orientation(lbgt_0,39,391). coords(lbgt_0,272,257,391).  visible(lbgt_0,391). happensAt(walking(lbgt_0),391). orientation(lbgt_4,147,391). coords(lbgt_4,80,71,391).  visible(lbgt_4,391). happensAt(walking(lbgt_4),391).
orientation(lbgt_0,47,392). coords(lbgt_0,273,258,392).  visible(lbgt_0,392). happensAt(walking(lbgt_0),392). orientation(lbgt_4,147,392). coords(lbgt_4,80,72,392).  visible(lbgt_4,392). happensAt(walking(lbgt_4),392).
orientation(lbgt_0,53,393). coords(lbgt_0,273,260,393).  visible(lbgt_0,393). happensAt(walking(lbgt_0),393). orientation(lbgt_4,147,393). coords(lbgt_4,80,72,393).  visible(lbgt_4,393). happensAt(walking(lbgt_4),393).
orientation(lbgt_0,53,394). coords(lbgt_0,273,262,394).  visible(lbgt_0,394). happensAt(walking(lbgt_0),394). orientation(lbgt_4,147,394). coords(lbgt_4,80,72,394).  visible(lbgt_4,394). happensAt(walking(lbgt_4),394).
orientation(lbgt_0,53,395). coords(lbgt_0,274,264,395).  visible(lbgt_0,395). happensAt(walking(lbgt_0),395). orientation(lbgt_4,147,395). coords(lbgt_4,80,73,395).  visible(lbgt_4,395). happensAt(walking(lbgt_4),395).
orientation(lbgt_0,53,396). coords(lbgt_0,274,267,396).  visible(lbgt_0,396). happensAt(walking(lbgt_0),396). orientation(lbgt_4,147,396). coords(lbgt_4,80,73,396).  visible(lbgt_4,396). happensAt(walking(lbgt_4),396).
orientation(lbgt_0,53,397). coords(lbgt_0,275,269,397).  visible(lbgt_0,397). happensAt(walking(lbgt_0),397). orientation(lbgt_4,147,397). coords(lbgt_4,80,74,397).  visible(lbgt_4,397). happensAt(walking(lbgt_4),397).
orientation(lbgt_0,53,398). coords(lbgt_0,276,272,398).  visible(lbgt_0,398). happensAt(walking(lbgt_0),398). orientation(lbgt_4,147,398). coords(lbgt_4,80,74,398).  visible(lbgt_4,398). happensAt(walking(lbgt_4),398).
orientation(lbgt_0,53,399). coords(lbgt_0,277,274,399).  visible(lbgt_0,399). happensAt(walking(lbgt_0),399). orientation(lbgt_4,147,399). coords(lbgt_4,80,74,399).  visible(lbgt_4,399). happensAt(walking(lbgt_4),399).
orientation(lbgt_0,53,400). coords(lbgt_0,277,278,400).  visible(lbgt_0,400). happensAt(walking(lbgt_0),400). orientation(lbgt_4,147,400). coords(lbgt_4,80,74,400).  visible(lbgt_4,400). happensAt(walking(lbgt_4),400).
orientation(lbgt_0,53,401). coords(lbgt_0,277,278,401).  visible(lbgt_0,401). happensAt(walking(lbgt_0),401). orientation(lbgt_4,147,401). coords(lbgt_4,80,74,401).  visible(lbgt_4,401). happensAt(walking(lbgt_4),401).
orientation(lbgt_0,53,402). coords(lbgt_0,277,278,402).  visible(lbgt_0,402). happensAt(walking(lbgt_0),402). orientation(lbgt_4,147,402). coords(lbgt_4,80,75,402).  visible(lbgt_4,402). happensAt(walking(lbgt_4),402).
orientation(lbgt_0,53,403). coords(lbgt_0,277,278,403).  visible(lbgt_0,403). happensAt(walking(lbgt_0),403). orientation(lbgt_4,147,403). coords(lbgt_4,80,75,403).  visible(lbgt_4,403). happensAt(walking(lbgt_4),403).
orientation(lbgt_0,53,404). coords(lbgt_0,280,278,404).  visible(lbgt_0,404). happensAt(walking(lbgt_0),404). orientation(lbgt_4,147,404). coords(lbgt_4,80,76,404).  visible(lbgt_4,404). happensAt(walking(lbgt_4),404).
orientation(lbgt_0,53,405). coords(lbgt_0,280,278,405).  visible(lbgt_0,405). happensAt(walking(lbgt_0),405). orientation(lbgt_4,147,405). coords(lbgt_4,80,76,405).  visible(lbgt_4,405). happensAt(walking(lbgt_4),405).
orientation(lbgt_0,53,406). coords(lbgt_0,280,278,406).  visible(lbgt_0,406). happensAt(walking(lbgt_0),406). orientation(lbgt_4,147,406). coords(lbgt_4,80,77,406).  visible(lbgt_4,406). happensAt(walking(lbgt_4),406).
orientation(lbgt_0,53,407). coords(lbgt_0,280,278,407).  visible(lbgt_0,407). happensAt(walking(lbgt_0),407). orientation(lbgt_4,147,407). coords(lbgt_4,80,77,407).  visible(lbgt_4,407). happensAt(walking(lbgt_4),407).
orientation(lbgt_0,53,408). coords(lbgt_0,280,279,408).  visible(lbgt_0,408). happensAt(walking(lbgt_0),408). orientation(lbgt_4,147,408). coords(lbgt_4,80,78,408).  visible(lbgt_4,408). happensAt(walking(lbgt_4),408).
orientation(lbgt_0,53,409). coords(lbgt_0,281,281,409).  visible(lbgt_0,409). happensAt(walking(lbgt_0),409). orientation(lbgt_4,147,409). coords(lbgt_4,80,78,409).  visible(lbgt_4,409). happensAt(walking(lbgt_4),409).
orientation(lbgt_0,53,410). coords(lbgt_0,282,283,410).  visible(lbgt_0,410). happensAt(walking(lbgt_0),410). orientation(lbgt_4,147,410). coords(lbgt_4,80,79,410).  visible(lbgt_4,410). happensAt(walking(lbgt_4),410).
orientation(lbgt_0,53,411). coords(lbgt_0,283,285,411).  visible(lbgt_0,411). happensAt(walking(lbgt_0),411). orientation(lbgt_4,147,411). coords(lbgt_4,80,79,411).  visible(lbgt_4,411). happensAt(walking(lbgt_4),411).
orientation(lbgt_0,53,412). coords(lbgt_0,283,285,412).  happensAt(disappear(lbgt_0),412). happensAt(walking(lbgt_0),412). orientation(lbgt_4,147,412). coords(lbgt_4,80,79,412).  visible(lbgt_4,412). happensAt(walking(lbgt_4),412).
#show holdsAt/2.