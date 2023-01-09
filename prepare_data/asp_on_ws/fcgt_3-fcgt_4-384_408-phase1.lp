
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

	orientation(fcgt_3,164,384). coords(fcgt_3,99,187,384).  visible(fcgt_3,384). happensAt(walking(fcgt_3),384). orientation(fcgt_4,166,384). coords(fcgt_4,82,206,384).  visible(fcgt_4,384). happensAt(inactive(fcgt_4),384).
orientation(fcgt_3,164,385). coords(fcgt_3,99,188,385).  visible(fcgt_3,385). happensAt(walking(fcgt_3),385). orientation(fcgt_4,166,385). coords(fcgt_4,82,206,385).  visible(fcgt_4,385). happensAt(inactive(fcgt_4),385).
orientation(fcgt_3,164,386). coords(fcgt_3,99,188,386).  visible(fcgt_3,386). happensAt(walking(fcgt_3),386). orientation(fcgt_4,166,386). coords(fcgt_4,82,206,386).  visible(fcgt_4,386). happensAt(inactive(fcgt_4),386).
orientation(fcgt_3,164,387). coords(fcgt_3,99,188,387).  visible(fcgt_3,387). happensAt(walking(fcgt_3),387). orientation(fcgt_4,166,387). coords(fcgt_4,82,206,387).  visible(fcgt_4,387). happensAt(inactive(fcgt_4),387).
orientation(fcgt_3,164,388). coords(fcgt_3,99,189,388).  visible(fcgt_3,388). happensAt(walking(fcgt_3),388). orientation(fcgt_4,166,388). coords(fcgt_4,82,206,388).  visible(fcgt_4,388). happensAt(inactive(fcgt_4),388).
orientation(fcgt_3,164,389). coords(fcgt_3,99,189,389).  visible(fcgt_3,389). happensAt(walking(fcgt_3),389). orientation(fcgt_4,166,389). coords(fcgt_4,82,206,389).  visible(fcgt_4,389). happensAt(inactive(fcgt_4),389).
orientation(fcgt_3,164,390). coords(fcgt_3,100,190,390).  visible(fcgt_3,390). happensAt(walking(fcgt_3),390). orientation(fcgt_4,166,390). coords(fcgt_4,82,206,390).  visible(fcgt_4,390). happensAt(inactive(fcgt_4),390).
orientation(fcgt_3,164,391). coords(fcgt_3,100,190,391).  visible(fcgt_3,391). happensAt(walking(fcgt_3),391). orientation(fcgt_4,166,391). coords(fcgt_4,82,206,391).  visible(fcgt_4,391). happensAt(inactive(fcgt_4),391).
orientation(fcgt_3,164,392). coords(fcgt_3,100,190,392).  visible(fcgt_3,392). happensAt(walking(fcgt_3),392). orientation(fcgt_4,166,392). coords(fcgt_4,82,206,392).  visible(fcgt_4,392). happensAt(inactive(fcgt_4),392).
orientation(fcgt_3,164,393). coords(fcgt_3,100,190,393).  visible(fcgt_3,393). happensAt(walking(fcgt_3),393). orientation(fcgt_4,166,393). coords(fcgt_4,82,206,393).  visible(fcgt_4,393). happensAt(inactive(fcgt_4),393).
orientation(fcgt_3,164,394). coords(fcgt_3,100,191,394).  visible(fcgt_3,394). happensAt(walking(fcgt_3),394). orientation(fcgt_4,166,394). coords(fcgt_4,82,206,394).  visible(fcgt_4,394). happensAt(inactive(fcgt_4),394).
orientation(fcgt_3,164,395). coords(fcgt_3,100,192,395).  visible(fcgt_3,395). happensAt(walking(fcgt_3),395). orientation(fcgt_4,166,395). coords(fcgt_4,82,206,395).  visible(fcgt_4,395). happensAt(inactive(fcgt_4),395).
orientation(fcgt_3,164,396). coords(fcgt_3,100,193,396).  visible(fcgt_3,396). happensAt(walking(fcgt_3),396). orientation(fcgt_4,166,396). coords(fcgt_4,82,206,396).  visible(fcgt_4,396). happensAt(inactive(fcgt_4),396).
orientation(fcgt_3,164,397). coords(fcgt_3,100,194,397).  visible(fcgt_3,397). happensAt(walking(fcgt_3),397). orientation(fcgt_4,166,397). coords(fcgt_4,82,206,397).  visible(fcgt_4,397). happensAt(inactive(fcgt_4),397).
orientation(fcgt_3,164,398). coords(fcgt_3,100,195,398).  visible(fcgt_3,398). happensAt(walking(fcgt_3),398). orientation(fcgt_4,166,398). coords(fcgt_4,82,206,398).  visible(fcgt_4,398). happensAt(inactive(fcgt_4),398).
orientation(fcgt_3,164,399). coords(fcgt_3,100,195,399).  visible(fcgt_3,399). happensAt(walking(fcgt_3),399). orientation(fcgt_4,166,399). coords(fcgt_4,82,206,399).  visible(fcgt_4,399). happensAt(inactive(fcgt_4),399).
orientation(fcgt_3,164,400). coords(fcgt_3,100,196,400).  visible(fcgt_3,400). happensAt(walking(fcgt_3),400). orientation(fcgt_4,166,400). coords(fcgt_4,82,206,400).  visible(fcgt_4,400). happensAt(inactive(fcgt_4),400).
orientation(fcgt_3,164,401). coords(fcgt_3,100,197,401).  visible(fcgt_3,401). happensAt(walking(fcgt_3),401). orientation(fcgt_4,166,401). coords(fcgt_4,82,206,401).  visible(fcgt_4,401). happensAt(inactive(fcgt_4),401).
orientation(fcgt_3,164,402). coords(fcgt_3,99,197,402).  visible(fcgt_3,402). happensAt(walking(fcgt_3),402). orientation(fcgt_4,166,402). coords(fcgt_4,82,206,402).  visible(fcgt_4,402). happensAt(inactive(fcgt_4),402).
orientation(fcgt_3,164,403). coords(fcgt_3,99,198,403).  visible(fcgt_3,403). happensAt(walking(fcgt_3),403). orientation(fcgt_4,166,403). coords(fcgt_4,82,206,403).  visible(fcgt_4,403). happensAt(inactive(fcgt_4),403).
orientation(fcgt_3,164,404). coords(fcgt_3,99,198,404).  visible(fcgt_3,404). happensAt(walking(fcgt_3),404). orientation(fcgt_4,166,404). coords(fcgt_4,82,206,404).  visible(fcgt_4,404). happensAt(inactive(fcgt_4),404).
orientation(fcgt_3,164,405). coords(fcgt_3,99,198,405).  visible(fcgt_3,405). happensAt(walking(fcgt_3),405). orientation(fcgt_4,166,405). coords(fcgt_4,82,206,405).  visible(fcgt_4,405). happensAt(inactive(fcgt_4),405).
orientation(fcgt_3,164,406). coords(fcgt_3,99,199,406).  visible(fcgt_3,406). happensAt(walking(fcgt_3),406). orientation(fcgt_4,166,406). coords(fcgt_4,82,206,406).  visible(fcgt_4,406). happensAt(inactive(fcgt_4),406).
orientation(fcgt_3,164,407). coords(fcgt_3,99,199,407).  visible(fcgt_3,407). happensAt(walking(fcgt_3),407). orientation(fcgt_4,166,407). coords(fcgt_4,82,206,407).  visible(fcgt_4,407). happensAt(inactive(fcgt_4),407).
orientation(fcgt_3,164,408). coords(fcgt_3,99,200,408).  visible(fcgt_3,408). happensAt(walking(fcgt_3),408). orientation(fcgt_4,166,408). coords(fcgt_4,82,206,408).  visible(fcgt_4,408). happensAt(inactive(fcgt_4),408).
#show holdsAt/2.