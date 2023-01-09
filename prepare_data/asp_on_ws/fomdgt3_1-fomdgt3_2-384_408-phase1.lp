
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

	orientation(fomdgt3_1,150,384). coords(fomdgt3_1,46,102,384).  visible(fomdgt3_1,384). happensAt(inactive(fomdgt3_1),384). orientation(fomdgt3_2,42,384). coords(fomdgt3_2,262,96,384).  visible(fomdgt3_2,384). happensAt(inactive(fomdgt3_2),384).
orientation(fomdgt3_1,150,385). coords(fomdgt3_1,46,102,385).  visible(fomdgt3_1,385). happensAt(inactive(fomdgt3_1),385). orientation(fomdgt3_2,42,385). coords(fomdgt3_2,262,96,385).  visible(fomdgt3_2,385). happensAt(inactive(fomdgt3_2),385).
orientation(fomdgt3_1,150,386). coords(fomdgt3_1,46,102,386).  visible(fomdgt3_1,386). happensAt(inactive(fomdgt3_1),386). orientation(fomdgt3_2,42,386). coords(fomdgt3_2,262,96,386).  visible(fomdgt3_2,386). happensAt(inactive(fomdgt3_2),386).
orientation(fomdgt3_1,150,387). coords(fomdgt3_1,46,102,387).  visible(fomdgt3_1,387). happensAt(inactive(fomdgt3_1),387). orientation(fomdgt3_2,42,387). coords(fomdgt3_2,262,96,387).  visible(fomdgt3_2,387). happensAt(inactive(fomdgt3_2),387).
orientation(fomdgt3_1,150,388). coords(fomdgt3_1,46,102,388).  visible(fomdgt3_1,388). happensAt(inactive(fomdgt3_1),388). orientation(fomdgt3_2,42,388). coords(fomdgt3_2,262,96,388).  visible(fomdgt3_2,388). happensAt(inactive(fomdgt3_2),388).
orientation(fomdgt3_1,150,389). coords(fomdgt3_1,46,102,389).  visible(fomdgt3_1,389). happensAt(inactive(fomdgt3_1),389). orientation(fomdgt3_2,42,389). coords(fomdgt3_2,262,96,389).  visible(fomdgt3_2,389). happensAt(inactive(fomdgt3_2),389).
orientation(fomdgt3_1,150,390). coords(fomdgt3_1,46,102,390).  visible(fomdgt3_1,390). happensAt(inactive(fomdgt3_1),390). orientation(fomdgt3_2,42,390). coords(fomdgt3_2,262,96,390).  visible(fomdgt3_2,390). happensAt(inactive(fomdgt3_2),390).
orientation(fomdgt3_1,150,391). coords(fomdgt3_1,46,102,391).  visible(fomdgt3_1,391). happensAt(inactive(fomdgt3_1),391). orientation(fomdgt3_2,42,391). coords(fomdgt3_2,262,96,391).  visible(fomdgt3_2,391). happensAt(inactive(fomdgt3_2),391).
orientation(fomdgt3_1,150,392). coords(fomdgt3_1,46,102,392).  visible(fomdgt3_1,392). happensAt(inactive(fomdgt3_1),392). orientation(fomdgt3_2,42,392). coords(fomdgt3_2,262,96,392).  visible(fomdgt3_2,392). happensAt(inactive(fomdgt3_2),392).
orientation(fomdgt3_1,150,393). coords(fomdgt3_1,46,102,393).  visible(fomdgt3_1,393). happensAt(inactive(fomdgt3_1),393). orientation(fomdgt3_2,42,393). coords(fomdgt3_2,262,96,393).  visible(fomdgt3_2,393). happensAt(inactive(fomdgt3_2),393).
orientation(fomdgt3_1,150,394). coords(fomdgt3_1,46,102,394).  visible(fomdgt3_1,394). happensAt(inactive(fomdgt3_1),394). orientation(fomdgt3_2,42,394). coords(fomdgt3_2,262,96,394).  visible(fomdgt3_2,394). happensAt(inactive(fomdgt3_2),394).
orientation(fomdgt3_1,150,395). coords(fomdgt3_1,46,102,395).  visible(fomdgt3_1,395). happensAt(inactive(fomdgt3_1),395). orientation(fomdgt3_2,42,395). coords(fomdgt3_2,262,96,395).  visible(fomdgt3_2,395). happensAt(inactive(fomdgt3_2),395).
orientation(fomdgt3_1,150,396). coords(fomdgt3_1,46,102,396).  visible(fomdgt3_1,396). happensAt(inactive(fomdgt3_1),396). orientation(fomdgt3_2,42,396). coords(fomdgt3_2,262,96,396).  visible(fomdgt3_2,396). happensAt(inactive(fomdgt3_2),396).
orientation(fomdgt3_1,150,397). coords(fomdgt3_1,46,102,397).  visible(fomdgt3_1,397). happensAt(inactive(fomdgt3_1),397). orientation(fomdgt3_2,42,397). coords(fomdgt3_2,262,96,397).  visible(fomdgt3_2,397). happensAt(inactive(fomdgt3_2),397).
orientation(fomdgt3_1,150,398). coords(fomdgt3_1,46,102,398).  visible(fomdgt3_1,398). happensAt(inactive(fomdgt3_1),398). orientation(fomdgt3_2,42,398). coords(fomdgt3_2,262,96,398).  visible(fomdgt3_2,398). happensAt(inactive(fomdgt3_2),398).
orientation(fomdgt3_1,150,399). coords(fomdgt3_1,46,102,399).  visible(fomdgt3_1,399). happensAt(inactive(fomdgt3_1),399). orientation(fomdgt3_2,42,399). coords(fomdgt3_2,262,96,399).  visible(fomdgt3_2,399). happensAt(inactive(fomdgt3_2),399).
orientation(fomdgt3_1,150,400). coords(fomdgt3_1,46,102,400).  visible(fomdgt3_1,400). happensAt(inactive(fomdgt3_1),400). orientation(fomdgt3_2,42,400). coords(fomdgt3_2,262,96,400).  visible(fomdgt3_2,400). happensAt(inactive(fomdgt3_2),400).
orientation(fomdgt3_1,150,401). coords(fomdgt3_1,46,102,401).  visible(fomdgt3_1,401). happensAt(inactive(fomdgt3_1),401). orientation(fomdgt3_2,42,401). coords(fomdgt3_2,262,96,401).  visible(fomdgt3_2,401). happensAt(inactive(fomdgt3_2),401).
orientation(fomdgt3_1,150,402). coords(fomdgt3_1,46,102,402).  visible(fomdgt3_1,402). happensAt(inactive(fomdgt3_1),402). orientation(fomdgt3_2,42,402). coords(fomdgt3_2,262,96,402).  visible(fomdgt3_2,402). happensAt(inactive(fomdgt3_2),402).
orientation(fomdgt3_1,150,403). coords(fomdgt3_1,46,102,403).  visible(fomdgt3_1,403). happensAt(inactive(fomdgt3_1),403). orientation(fomdgt3_2,42,403). coords(fomdgt3_2,262,96,403).  visible(fomdgt3_2,403). happensAt(inactive(fomdgt3_2),403).
orientation(fomdgt3_1,150,404). coords(fomdgt3_1,46,102,404).  visible(fomdgt3_1,404). happensAt(inactive(fomdgt3_1),404). orientation(fomdgt3_2,42,404). coords(fomdgt3_2,262,96,404).  visible(fomdgt3_2,404). happensAt(inactive(fomdgt3_2),404).
orientation(fomdgt3_1,150,405). coords(fomdgt3_1,46,102,405).  visible(fomdgt3_1,405). happensAt(inactive(fomdgt3_1),405). orientation(fomdgt3_2,42,405). coords(fomdgt3_2,262,96,405).  visible(fomdgt3_2,405). happensAt(inactive(fomdgt3_2),405).
orientation(fomdgt3_1,150,406). coords(fomdgt3_1,46,102,406).  visible(fomdgt3_1,406). happensAt(inactive(fomdgt3_1),406). orientation(fomdgt3_2,42,406). coords(fomdgt3_2,262,96,406).  visible(fomdgt3_2,406). happensAt(inactive(fomdgt3_2),406).
orientation(fomdgt3_1,150,407). coords(fomdgt3_1,46,102,407).  visible(fomdgt3_1,407). happensAt(inactive(fomdgt3_1),407). orientation(fomdgt3_2,42,407). coords(fomdgt3_2,262,96,407).  visible(fomdgt3_2,407). happensAt(inactive(fomdgt3_2),407).
orientation(fomdgt3_1,150,408). coords(fomdgt3_1,46,102,408).  visible(fomdgt3_1,408). happensAt(inactive(fomdgt3_1),408). orientation(fomdgt3_2,42,408). coords(fomdgt3_2,262,96,408).  visible(fomdgt3_2,408). happensAt(inactive(fomdgt3_2),408).
#show holdsAt/2.