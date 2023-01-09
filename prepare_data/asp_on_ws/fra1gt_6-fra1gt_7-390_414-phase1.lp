
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

	orientation(fra1gt_6,160,390). coords(fra1gt_6,141,99,390).  visible(fra1gt_6,390). happensAt(running(fra1gt_6),390). orientation(fra1gt_7,162,390). coords(fra1gt_7,129,74,390).  visible(fra1gt_7,390). happensAt(running(fra1gt_7),390).
orientation(fra1gt_6,160,391). coords(fra1gt_6,139,98,391).  visible(fra1gt_6,391). happensAt(running(fra1gt_6),391). orientation(fra1gt_7,162,391). coords(fra1gt_7,126,75,391).  visible(fra1gt_7,391). happensAt(running(fra1gt_7),391).
orientation(fra1gt_6,160,392). coords(fra1gt_6,138,98,392).  visible(fra1gt_6,392). happensAt(running(fra1gt_6),392). orientation(fra1gt_7,162,392). coords(fra1gt_7,123,75,392).  visible(fra1gt_7,392). happensAt(running(fra1gt_7),392).
orientation(fra1gt_6,160,393). coords(fra1gt_6,135,98,393).  visible(fra1gt_6,393). happensAt(running(fra1gt_6),393). orientation(fra1gt_7,162,393). coords(fra1gt_7,120,75,393).  visible(fra1gt_7,393). happensAt(running(fra1gt_7),393).
orientation(fra1gt_6,160,394). coords(fra1gt_6,135,99,394).  visible(fra1gt_6,394). happensAt(running(fra1gt_6),394). orientation(fra1gt_7,162,394). coords(fra1gt_7,117,75,394).  visible(fra1gt_7,394). happensAt(running(fra1gt_7),394).
orientation(fra1gt_6,160,395). coords(fra1gt_6,130,101,395).  visible(fra1gt_6,395). happensAt(running(fra1gt_6),395). orientation(fra1gt_7,162,395). coords(fra1gt_7,115,75,395).  visible(fra1gt_7,395). happensAt(running(fra1gt_7),395).
orientation(fra1gt_6,160,396). coords(fra1gt_6,127,102,396).  visible(fra1gt_6,396). happensAt(running(fra1gt_6),396). orientation(fra1gt_7,162,396). coords(fra1gt_7,113,76,396).  visible(fra1gt_7,396). happensAt(running(fra1gt_7),396).
orientation(fra1gt_6,160,397). coords(fra1gt_6,127,104,397).  visible(fra1gt_6,397). happensAt(running(fra1gt_6),397). orientation(fra1gt_7,162,397). coords(fra1gt_7,112,75,397).  visible(fra1gt_7,397). happensAt(running(fra1gt_7),397).
orientation(fra1gt_6,160,398). coords(fra1gt_6,125,106,398).  visible(fra1gt_6,398). happensAt(running(fra1gt_6),398). orientation(fra1gt_7,162,398). coords(fra1gt_7,110,75,398).  visible(fra1gt_7,398). happensAt(running(fra1gt_7),398).
orientation(fra1gt_6,160,399). coords(fra1gt_6,123,108,399).  visible(fra1gt_6,399). happensAt(running(fra1gt_6),399). orientation(fra1gt_7,158,399). coords(fra1gt_7,108,76,399).  visible(fra1gt_7,399). happensAt(running(fra1gt_7),399).
orientation(fra1gt_6,160,400). coords(fra1gt_6,122,110,400).  visible(fra1gt_6,400). happensAt(running(fra1gt_6),400). orientation(fra1gt_7,155,400). coords(fra1gt_7,105,76,400).  visible(fra1gt_7,400). happensAt(running(fra1gt_7),400).
orientation(fra1gt_6,160,401). coords(fra1gt_6,119,112,401).  visible(fra1gt_6,401). happensAt(running(fra1gt_6),401). orientation(fra1gt_7,146,401). coords(fra1gt_7,103,77,401).  visible(fra1gt_7,401). happensAt(running(fra1gt_7),401).
orientation(fra1gt_6,160,402). coords(fra1gt_6,117,113,402).  visible(fra1gt_6,402). happensAt(running(fra1gt_6),402). orientation(fra1gt_7,146,402). coords(fra1gt_7,102,78,402).  visible(fra1gt_7,402). happensAt(running(fra1gt_7),402).
orientation(fra1gt_6,160,403). coords(fra1gt_6,115,115,403).  visible(fra1gt_6,403). happensAt(running(fra1gt_6),403). orientation(fra1gt_7,146,403). coords(fra1gt_7,100,78,403).  visible(fra1gt_7,403). happensAt(running(fra1gt_7),403).
orientation(fra1gt_6,160,404). coords(fra1gt_6,111,115,404).  visible(fra1gt_6,404). happensAt(running(fra1gt_6),404). orientation(fra1gt_7,146,404). coords(fra1gt_7,99,78,404).  visible(fra1gt_7,404). happensAt(running(fra1gt_7),404).
orientation(fra1gt_6,160,405). coords(fra1gt_6,108,115,405).  visible(fra1gt_6,405). happensAt(running(fra1gt_6),405). orientation(fra1gt_7,146,405). coords(fra1gt_7,98,78,405).  visible(fra1gt_7,405). happensAt(running(fra1gt_7),405).
orientation(fra1gt_6,164,406). coords(fra1gt_6,106,116,406).  visible(fra1gt_6,406). happensAt(running(fra1gt_6),406). orientation(fra1gt_7,146,406). coords(fra1gt_7,96,78,406).  visible(fra1gt_7,406). happensAt(running(fra1gt_7),406).
orientation(fra1gt_6,167,407). coords(fra1gt_6,105,116,407).  visible(fra1gt_6,407). happensAt(running(fra1gt_6),407). orientation(fra1gt_7,146,407). coords(fra1gt_7,94,78,407).  visible(fra1gt_7,407). happensAt(running(fra1gt_7),407).
orientation(fra1gt_6,167,408). coords(fra1gt_6,103,116,408).  visible(fra1gt_6,408). happensAt(running(fra1gt_6),408). orientation(fra1gt_7,146,408). coords(fra1gt_7,93,79,408).  visible(fra1gt_7,408). happensAt(running(fra1gt_7),408).
orientation(fra1gt_6,167,409). coords(fra1gt_6,101,117,409).  visible(fra1gt_6,409). happensAt(running(fra1gt_6),409). orientation(fra1gt_7,146,409). coords(fra1gt_7,91,79,409).  visible(fra1gt_7,409). happensAt(running(fra1gt_7),409).
orientation(fra1gt_6,167,410). coords(fra1gt_6,100,121,410).  visible(fra1gt_6,410). happensAt(running(fra1gt_6),410). orientation(fra1gt_7,146,410). coords(fra1gt_7,89,79,410).  visible(fra1gt_7,410). happensAt(running(fra1gt_7),410).
orientation(fra1gt_6,167,411). coords(fra1gt_6,97,123,411).  visible(fra1gt_6,411). happensAt(running(fra1gt_6),411). orientation(fra1gt_7,146,411). coords(fra1gt_7,88,79,411).  visible(fra1gt_7,411). happensAt(running(fra1gt_7),411).
orientation(fra1gt_6,173,412). coords(fra1gt_6,95,125,412).  visible(fra1gt_6,412). happensAt(running(fra1gt_6),412). orientation(fra1gt_7,146,412). coords(fra1gt_7,88,79,412).  visible(fra1gt_7,412). happensAt(running(fra1gt_7),412).
orientation(fra1gt_6,173,413). coords(fra1gt_6,92,126,413).  visible(fra1gt_6,413). happensAt(running(fra1gt_6),413). orientation(fra1gt_7,146,413). coords(fra1gt_7,87,78,413).  visible(fra1gt_7,413). happensAt(running(fra1gt_7),413).
orientation(fra1gt_6,173,414). coords(fra1gt_6,90,128,414).  visible(fra1gt_6,414). happensAt(running(fra1gt_6),414). orientation(fra1gt_7,146,414). coords(fra1gt_7,85,79,414).  visible(fra1gt_7,414). happensAt(running(fra1gt_7),414).
#show holdsAt/2.