
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

	orientation(bww1gt_1,179,386). coords(bww1gt_1,105,264,386).  visible(bww1gt_1,386). happensAt(inactive(bww1gt_1),386). orientation(bww1gt_2,158,386). coords(bww1gt_2,104,149,386).  visible(bww1gt_2,386). happensAt(walking(bww1gt_2),386).
orientation(bww1gt_1,179,387). coords(bww1gt_1,105,264,387).  visible(bww1gt_1,387). happensAt(inactive(bww1gt_1),387). orientation(bww1gt_2,157,387). coords(bww1gt_2,105,149,387).  visible(bww1gt_2,387). happensAt(walking(bww1gt_2),387).
orientation(bww1gt_1,2,388). coords(bww1gt_1,105,264,388).  visible(bww1gt_1,388). happensAt(inactive(bww1gt_1),388). orientation(bww1gt_2,160,388). coords(bww1gt_2,106,149,388).  visible(bww1gt_2,388). happensAt(walking(bww1gt_2),388).
orientation(bww1gt_1,3,389). coords(bww1gt_1,105,264,389).  visible(bww1gt_1,389). happensAt(inactive(bww1gt_1),389). orientation(bww1gt_2,157,389). coords(bww1gt_2,106,148,389).  visible(bww1gt_2,389). happensAt(walking(bww1gt_2),389).
orientation(bww1gt_1,1,390). coords(bww1gt_1,105,264,390).  visible(bww1gt_1,390). happensAt(inactive(bww1gt_1),390). orientation(bww1gt_2,154,390). coords(bww1gt_2,106,148,390).  visible(bww1gt_2,390). happensAt(walking(bww1gt_2),390).
orientation(bww1gt_1,2,391). coords(bww1gt_1,105,264,391).  visible(bww1gt_1,391). happensAt(inactive(bww1gt_1),391). orientation(bww1gt_2,158,391). coords(bww1gt_2,107,147,391).  visible(bww1gt_2,391). happensAt(walking(bww1gt_2),391).
orientation(bww1gt_1,0,392). coords(bww1gt_1,105,264,392).  visible(bww1gt_1,392). happensAt(inactive(bww1gt_1),392). orientation(bww1gt_2,157,392). coords(bww1gt_2,108,147,392).  visible(bww1gt_2,392). happensAt(walking(bww1gt_2),392).
orientation(bww1gt_1,3,393). coords(bww1gt_1,105,264,393).  visible(bww1gt_1,393). happensAt(inactive(bww1gt_1),393). orientation(bww1gt_2,160,393). coords(bww1gt_2,108,147,393).  visible(bww1gt_2,393). happensAt(walking(bww1gt_2),393).
orientation(bww1gt_1,7,394). coords(bww1gt_1,105,264,394).  visible(bww1gt_1,394). happensAt(inactive(bww1gt_1),394). orientation(bww1gt_2,156,394). coords(bww1gt_2,108,146,394).  visible(bww1gt_2,394). happensAt(walking(bww1gt_2),394).
orientation(bww1gt_1,1,395). coords(bww1gt_1,105,264,395).  visible(bww1gt_1,395). happensAt(inactive(bww1gt_1),395). orientation(bww1gt_2,158,395). coords(bww1gt_2,109,146,395).  visible(bww1gt_2,395). happensAt(walking(bww1gt_2),395).
orientation(bww1gt_1,2,396). coords(bww1gt_1,105,264,396).  visible(bww1gt_1,396). happensAt(inactive(bww1gt_1),396). orientation(bww1gt_2,157,396). coords(bww1gt_2,109,146,396).  visible(bww1gt_2,396). happensAt(walking(bww1gt_2),396).
orientation(bww1gt_1,3,397). coords(bww1gt_1,105,264,397).  visible(bww1gt_1,397). happensAt(inactive(bww1gt_1),397). orientation(bww1gt_2,156,397). coords(bww1gt_2,109,146,397).  visible(bww1gt_2,397). happensAt(walking(bww1gt_2),397).
orientation(bww1gt_1,179,398). coords(bww1gt_1,105,264,398).  visible(bww1gt_1,398). happensAt(inactive(bww1gt_1),398). orientation(bww1gt_2,155,398). coords(bww1gt_2,110,146,398).  visible(bww1gt_2,398). happensAt(walking(bww1gt_2),398).
orientation(bww1gt_1,1,399). coords(bww1gt_1,105,264,399).  visible(bww1gt_1,399). happensAt(inactive(bww1gt_1),399). orientation(bww1gt_2,154,399). coords(bww1gt_2,112,146,399).  visible(bww1gt_2,399). happensAt(walking(bww1gt_2),399).
orientation(bww1gt_1,4,400). coords(bww1gt_1,105,264,400).  visible(bww1gt_1,400). happensAt(inactive(bww1gt_1),400). orientation(bww1gt_2,158,400). coords(bww1gt_2,114,146,400).  visible(bww1gt_2,400). happensAt(walking(bww1gt_2),400).
orientation(bww1gt_1,1,401). coords(bww1gt_1,105,264,401).  visible(bww1gt_1,401). happensAt(inactive(bww1gt_1),401). orientation(bww1gt_2,158,401). coords(bww1gt_2,115,146,401).  visible(bww1gt_2,401). happensAt(walking(bww1gt_2),401).
orientation(bww1gt_1,4,402). coords(bww1gt_1,105,264,402).  visible(bww1gt_1,402). happensAt(inactive(bww1gt_1),402). orientation(bww1gt_2,157,402). coords(bww1gt_2,115,146,402).  visible(bww1gt_2,402). happensAt(walking(bww1gt_2),402).
orientation(bww1gt_1,5,403). coords(bww1gt_1,105,264,403).  visible(bww1gt_1,403). happensAt(inactive(bww1gt_1),403). orientation(bww1gt_2,159,403). coords(bww1gt_2,116,146,403).  visible(bww1gt_2,403). happensAt(walking(bww1gt_2),403).
orientation(bww1gt_1,0,404). coords(bww1gt_1,105,264,404).  visible(bww1gt_1,404). happensAt(inactive(bww1gt_1),404). orientation(bww1gt_2,158,404). coords(bww1gt_2,117,146,404).  visible(bww1gt_2,404). happensAt(walking(bww1gt_2),404).
orientation(bww1gt_1,3,405). coords(bww1gt_1,105,264,405).  visible(bww1gt_1,405). happensAt(inactive(bww1gt_1),405). orientation(bww1gt_2,158,405). coords(bww1gt_2,118,146,405).  visible(bww1gt_2,405). happensAt(walking(bww1gt_2),405).
orientation(bww1gt_1,4,406). coords(bww1gt_1,105,264,406).  visible(bww1gt_1,406). happensAt(inactive(bww1gt_1),406). orientation(bww1gt_2,158,406). coords(bww1gt_2,118,146,406).  visible(bww1gt_2,406). happensAt(walking(bww1gt_2),406).
orientation(bww1gt_1,8,407). coords(bww1gt_1,105,264,407).  visible(bww1gt_1,407). happensAt(inactive(bww1gt_1),407). orientation(bww1gt_2,158,407). coords(bww1gt_2,119,146,407).  visible(bww1gt_2,407). happensAt(walking(bww1gt_2),407).
orientation(bww1gt_1,179,408). coords(bww1gt_1,105,264,408).  visible(bww1gt_1,408). happensAt(inactive(bww1gt_1),408). orientation(bww1gt_2,157,408). coords(bww1gt_2,119,146,408).  visible(bww1gt_2,408). happensAt(walking(bww1gt_2),408).
orientation(bww1gt_1,2,409). coords(bww1gt_1,105,264,409).  visible(bww1gt_1,409). happensAt(inactive(bww1gt_1),409). orientation(bww1gt_2,156,409). coords(bww1gt_2,119,146,409).  visible(bww1gt_2,409). happensAt(walking(bww1gt_2),409).
orientation(bww1gt_1,2,410). coords(bww1gt_1,105,264,410).  visible(bww1gt_1,410). happensAt(inactive(bww1gt_1),410). orientation(bww1gt_2,160,410). coords(bww1gt_2,120,146,410).  visible(bww1gt_2,410). happensAt(walking(bww1gt_2),410).
#show holdsAt/2.