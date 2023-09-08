
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

	orientation(lb1gt_1,151,384). coords(lb1gt_1,166,168,384).  visible(lb1gt_1,384). happensAt(active(lb1gt_1),384). orientation(lb1gt_2,144,384). coords(lb1gt_2,190,141,384).  visible(lb1gt_2,384). happensAt(active(lb1gt_2),384).
orientation(lb1gt_1,151,385). coords(lb1gt_1,166,168,385).  visible(lb1gt_1,385). happensAt(active(lb1gt_1),385). orientation(lb1gt_2,144,385). coords(lb1gt_2,190,141,385).  visible(lb1gt_2,385). happensAt(active(lb1gt_2),385).
orientation(lb1gt_1,151,386). coords(lb1gt_1,166,168,386).  visible(lb1gt_1,386). happensAt(active(lb1gt_1),386). orientation(lb1gt_2,144,386). coords(lb1gt_2,190,141,386).  visible(lb1gt_2,386). happensAt(active(lb1gt_2),386).
orientation(lb1gt_1,151,387). coords(lb1gt_1,166,168,387).  visible(lb1gt_1,387). happensAt(active(lb1gt_1),387). orientation(lb1gt_2,144,387). coords(lb1gt_2,190,141,387).  visible(lb1gt_2,387). happensAt(active(lb1gt_2),387).
orientation(lb1gt_1,151,388). coords(lb1gt_1,166,168,388).  visible(lb1gt_1,388). happensAt(active(lb1gt_1),388). orientation(lb1gt_2,144,388). coords(lb1gt_2,190,141,388).  visible(lb1gt_2,388). happensAt(active(lb1gt_2),388).
orientation(lb1gt_1,151,389). coords(lb1gt_1,166,168,389).  visible(lb1gt_1,389). happensAt(active(lb1gt_1),389). orientation(lb1gt_2,144,389). coords(lb1gt_2,190,141,389).  visible(lb1gt_2,389). happensAt(active(lb1gt_2),389).
orientation(lb1gt_1,151,390). coords(lb1gt_1,166,169,390).  visible(lb1gt_1,390). happensAt(active(lb1gt_1),390). orientation(lb1gt_2,144,390). coords(lb1gt_2,190,141,390).  visible(lb1gt_2,390). happensAt(active(lb1gt_2),390).
orientation(lb1gt_1,151,391). coords(lb1gt_1,166,170,391).  visible(lb1gt_1,391). happensAt(active(lb1gt_1),391). orientation(lb1gt_2,144,391). coords(lb1gt_2,190,141,391).  visible(lb1gt_2,391). happensAt(active(lb1gt_2),391).
orientation(lb1gt_1,151,392). coords(lb1gt_1,166,171,392).  visible(lb1gt_1,392). happensAt(active(lb1gt_1),392). orientation(lb1gt_2,144,392). coords(lb1gt_2,190,141,392).  visible(lb1gt_2,392). happensAt(active(lb1gt_2),392).
orientation(lb1gt_1,151,393). coords(lb1gt_1,166,171,393).  visible(lb1gt_1,393). happensAt(active(lb1gt_1),393). orientation(lb1gt_2,144,393). coords(lb1gt_2,190,141,393).  visible(lb1gt_2,393). happensAt(active(lb1gt_2),393).
orientation(lb1gt_1,151,394). coords(lb1gt_1,166,171,394).  visible(lb1gt_1,394). happensAt(active(lb1gt_1),394). orientation(lb1gt_2,144,394). coords(lb1gt_2,190,141,394).  visible(lb1gt_2,394). happensAt(active(lb1gt_2),394).
orientation(lb1gt_1,151,395). coords(lb1gt_1,166,171,395).  visible(lb1gt_1,395). happensAt(active(lb1gt_1),395). orientation(lb1gt_2,144,395). coords(lb1gt_2,190,141,395).  visible(lb1gt_2,395). happensAt(active(lb1gt_2),395).
orientation(lb1gt_1,151,396). coords(lb1gt_1,166,170,396).  visible(lb1gt_1,396). happensAt(active(lb1gt_1),396). orientation(lb1gt_2,144,396). coords(lb1gt_2,189,141,396).  visible(lb1gt_2,396). happensAt(active(lb1gt_2),396).
orientation(lb1gt_1,151,397). coords(lb1gt_1,166,169,397).  visible(lb1gt_1,397). happensAt(active(lb1gt_1),397). orientation(lb1gt_2,144,397). coords(lb1gt_2,189,141,397).  visible(lb1gt_2,397). happensAt(active(lb1gt_2),397).
orientation(lb1gt_1,151,398). coords(lb1gt_1,166,169,398).  visible(lb1gt_1,398). happensAt(active(lb1gt_1),398). orientation(lb1gt_2,144,398). coords(lb1gt_2,189,141,398).  visible(lb1gt_2,398). happensAt(active(lb1gt_2),398).
orientation(lb1gt_1,151,399). coords(lb1gt_1,166,168,399).  visible(lb1gt_1,399). happensAt(active(lb1gt_1),399). orientation(lb1gt_2,144,399). coords(lb1gt_2,189,141,399).  visible(lb1gt_2,399). happensAt(active(lb1gt_2),399).
orientation(lb1gt_1,151,400). coords(lb1gt_1,166,168,400).  visible(lb1gt_1,400). happensAt(active(lb1gt_1),400). orientation(lb1gt_2,144,400). coords(lb1gt_2,189,141,400).  visible(lb1gt_2,400). happensAt(active(lb1gt_2),400).
orientation(lb1gt_1,151,401). coords(lb1gt_1,166,168,401).  visible(lb1gt_1,401). happensAt(active(lb1gt_1),401). orientation(lb1gt_2,144,401). coords(lb1gt_2,189,141,401).  visible(lb1gt_2,401). happensAt(active(lb1gt_2),401).
orientation(lb1gt_1,151,402). coords(lb1gt_1,166,168,402).  visible(lb1gt_1,402). happensAt(active(lb1gt_1),402). orientation(lb1gt_2,144,402). coords(lb1gt_2,189,141,402).  visible(lb1gt_2,402). happensAt(active(lb1gt_2),402).
orientation(lb1gt_1,151,403). coords(lb1gt_1,166,168,403).  visible(lb1gt_1,403). happensAt(active(lb1gt_1),403). orientation(lb1gt_2,144,403). coords(lb1gt_2,189,141,403).  visible(lb1gt_2,403). happensAt(active(lb1gt_2),403).
orientation(lb1gt_1,151,404). coords(lb1gt_1,166,168,404).  visible(lb1gt_1,404). happensAt(active(lb1gt_1),404). orientation(lb1gt_2,144,404). coords(lb1gt_2,189,141,404).  visible(lb1gt_2,404). happensAt(active(lb1gt_2),404).
orientation(lb1gt_1,151,405). coords(lb1gt_1,166,168,405).  visible(lb1gt_1,405). happensAt(active(lb1gt_1),405). orientation(lb1gt_2,144,405). coords(lb1gt_2,189,141,405).  visible(lb1gt_2,405). happensAt(active(lb1gt_2),405).
orientation(lb1gt_1,151,406). coords(lb1gt_1,166,168,406).  visible(lb1gt_1,406). happensAt(active(lb1gt_1),406). orientation(lb1gt_2,144,406). coords(lb1gt_2,189,141,406).  visible(lb1gt_2,406). happensAt(active(lb1gt_2),406).
orientation(lb1gt_1,151,407). coords(lb1gt_1,166,168,407).  visible(lb1gt_1,407). happensAt(active(lb1gt_1),407). orientation(lb1gt_2,144,407). coords(lb1gt_2,189,141,407).  visible(lb1gt_2,407). happensAt(active(lb1gt_2),407).
orientation(lb1gt_1,151,408). coords(lb1gt_1,166,168,408).  visible(lb1gt_1,408). happensAt(active(lb1gt_1),408). orientation(lb1gt_2,144,408). coords(lb1gt_2,189,141,408).  visible(lb1gt_2,408). happensAt(active(lb1gt_2),408).
#show holdsAt/2.