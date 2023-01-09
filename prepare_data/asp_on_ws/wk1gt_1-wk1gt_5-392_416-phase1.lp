
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

	orientation(wk1gt_1,147,392). coords(wk1gt_1,110,111,392).  visible(wk1gt_1,392). happensAt(walking(wk1gt_1),392). orientation(wk1gt_5,162,392). coords(wk1gt_5,50,163,392).  visible(wk1gt_5,392). happensAt(inactive(wk1gt_5),392).
orientation(wk1gt_1,153,393). coords(wk1gt_1,109,110,393).  visible(wk1gt_1,393). happensAt(walking(wk1gt_1),393). orientation(wk1gt_5,162,393). coords(wk1gt_5,50,163,393).  visible(wk1gt_5,393). happensAt(inactive(wk1gt_5),393).
orientation(wk1gt_1,147,394). coords(wk1gt_1,108,110,394).  visible(wk1gt_1,394). happensAt(walking(wk1gt_1),394). orientation(wk1gt_5,162,394). coords(wk1gt_5,50,163,394).  visible(wk1gt_5,394). happensAt(inactive(wk1gt_5),394).
orientation(wk1gt_1,147,395). coords(wk1gt_1,107,109,395).  visible(wk1gt_1,395). happensAt(walking(wk1gt_1),395). orientation(wk1gt_5,162,395). coords(wk1gt_5,50,163,395).  visible(wk1gt_5,395). happensAt(inactive(wk1gt_5),395).
orientation(wk1gt_1,147,396). coords(wk1gt_1,106,109,396).  visible(wk1gt_1,396). happensAt(walking(wk1gt_1),396). orientation(wk1gt_5,162,396). coords(wk1gt_5,50,163,396).  visible(wk1gt_5,396). happensAt(inactive(wk1gt_5),396).
orientation(wk1gt_1,154,397). coords(wk1gt_1,105,109,397).  visible(wk1gt_1,397). happensAt(walking(wk1gt_1),397). orientation(wk1gt_5,162,397). coords(wk1gt_5,50,163,397).  visible(wk1gt_5,397). happensAt(inactive(wk1gt_5),397).
orientation(wk1gt_1,154,398). coords(wk1gt_1,104,109,398).  visible(wk1gt_1,398). happensAt(walking(wk1gt_1),398). orientation(wk1gt_5,162,398). coords(wk1gt_5,50,163,398).  visible(wk1gt_5,398). happensAt(inactive(wk1gt_5),398).
orientation(wk1gt_1,154,399). coords(wk1gt_1,104,108,399).  visible(wk1gt_1,399). happensAt(walking(wk1gt_1),399). orientation(wk1gt_5,162,399). coords(wk1gt_5,50,163,399).  visible(wk1gt_5,399). happensAt(inactive(wk1gt_5),399).
orientation(wk1gt_1,154,400). coords(wk1gt_1,103,108,400).  visible(wk1gt_1,400). happensAt(walking(wk1gt_1),400). orientation(wk1gt_5,162,400). coords(wk1gt_5,50,163,400).  visible(wk1gt_5,400). happensAt(inactive(wk1gt_5),400).
orientation(wk1gt_1,154,401). coords(wk1gt_1,104,108,401).  visible(wk1gt_1,401). happensAt(walking(wk1gt_1),401). orientation(wk1gt_5,162,401). coords(wk1gt_5,50,163,401).  visible(wk1gt_5,401). happensAt(inactive(wk1gt_5),401).
orientation(wk1gt_1,154,402). coords(wk1gt_1,103,108,402).  visible(wk1gt_1,402). happensAt(walking(wk1gt_1),402). orientation(wk1gt_5,162,402). coords(wk1gt_5,50,163,402).  visible(wk1gt_5,402). happensAt(inactive(wk1gt_5),402).
orientation(wk1gt_1,148,403). coords(wk1gt_1,103,108,403).  visible(wk1gt_1,403). happensAt(walking(wk1gt_1),403). orientation(wk1gt_5,162,403). coords(wk1gt_5,50,163,403).  visible(wk1gt_5,403). happensAt(inactive(wk1gt_5),403).
orientation(wk1gt_1,145,404). coords(wk1gt_1,103,107,404).  visible(wk1gt_1,404). happensAt(walking(wk1gt_1),404). orientation(wk1gt_5,162,404). coords(wk1gt_5,50,163,404).  visible(wk1gt_5,404). happensAt(inactive(wk1gt_5),404).
orientation(wk1gt_1,146,405). coords(wk1gt_1,103,107,405).  visible(wk1gt_1,405). happensAt(walking(wk1gt_1),405). orientation(wk1gt_5,162,405). coords(wk1gt_5,50,163,405).  visible(wk1gt_5,405). happensAt(inactive(wk1gt_5),405).
orientation(wk1gt_1,145,406). coords(wk1gt_1,103,107,406).  visible(wk1gt_1,406). happensAt(walking(wk1gt_1),406). orientation(wk1gt_5,162,406). coords(wk1gt_5,50,163,406).  visible(wk1gt_5,406). happensAt(inactive(wk1gt_5),406).
orientation(wk1gt_1,143,407). coords(wk1gt_1,102,106,407).  visible(wk1gt_1,407). happensAt(walking(wk1gt_1),407). orientation(wk1gt_5,162,407). coords(wk1gt_5,50,163,407).  visible(wk1gt_5,407). happensAt(inactive(wk1gt_5),407).
orientation(wk1gt_1,146,408). coords(wk1gt_1,100,105,408).  visible(wk1gt_1,408). happensAt(walking(wk1gt_1),408). orientation(wk1gt_5,162,408). coords(wk1gt_5,50,163,408).  visible(wk1gt_5,408). happensAt(inactive(wk1gt_5),408).
orientation(wk1gt_1,146,409). coords(wk1gt_1,100,105,409).  visible(wk1gt_1,409). happensAt(walking(wk1gt_1),409). orientation(wk1gt_5,162,409). coords(wk1gt_5,50,163,409).  visible(wk1gt_5,409). happensAt(inactive(wk1gt_5),409).
orientation(wk1gt_1,146,410). coords(wk1gt_1,99,104,410).  visible(wk1gt_1,410). happensAt(walking(wk1gt_1),410). orientation(wk1gt_5,162,410). coords(wk1gt_5,50,163,410).  visible(wk1gt_5,410). happensAt(inactive(wk1gt_5),410).
orientation(wk1gt_1,146,411). coords(wk1gt_1,99,103,411).  visible(wk1gt_1,411). happensAt(walking(wk1gt_1),411). orientation(wk1gt_5,162,411). coords(wk1gt_5,50,163,411).  visible(wk1gt_5,411). happensAt(inactive(wk1gt_5),411).
orientation(wk1gt_1,146,412). coords(wk1gt_1,99,103,412).  visible(wk1gt_1,412). happensAt(walking(wk1gt_1),412). orientation(wk1gt_5,162,412). coords(wk1gt_5,50,163,412).  visible(wk1gt_5,412). happensAt(inactive(wk1gt_5),412).
orientation(wk1gt_1,146,413). coords(wk1gt_1,99,102,413).  visible(wk1gt_1,413). happensAt(walking(wk1gt_1),413). orientation(wk1gt_5,162,413). coords(wk1gt_5,50,163,413).  visible(wk1gt_5,413). happensAt(inactive(wk1gt_5),413).
orientation(wk1gt_1,146,414). coords(wk1gt_1,99,102,414).  visible(wk1gt_1,414). happensAt(walking(wk1gt_1),414). orientation(wk1gt_5,162,414). coords(wk1gt_5,50,163,414).  visible(wk1gt_5,414). happensAt(inactive(wk1gt_5),414).
orientation(wk1gt_1,146,415). coords(wk1gt_1,98,101,415).  visible(wk1gt_1,415). happensAt(walking(wk1gt_1),415). orientation(wk1gt_5,162,415). coords(wk1gt_5,50,163,415).  visible(wk1gt_5,415). happensAt(active(wk1gt_5),415).
orientation(wk1gt_1,146,416). coords(wk1gt_1,98,101,416).  visible(wk1gt_1,416). happensAt(walking(wk1gt_1),416). orientation(wk1gt_5,162,416). coords(wk1gt_5,50,163,416).  visible(wk1gt_5,416). happensAt(active(wk1gt_5),416).
#show holdsAt/2.