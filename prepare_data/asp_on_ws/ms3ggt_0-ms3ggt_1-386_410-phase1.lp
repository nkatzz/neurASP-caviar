
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

	orientation(ms3ggt_0,152,386). coords(ms3ggt_0,142,140,386).  visible(ms3ggt_0,386). happensAt(active(ms3ggt_0),386). orientation(ms3ggt_1,162,386). coords(ms3ggt_1,142,157,386).  visible(ms3ggt_1,386). happensAt(walking(ms3ggt_1),386).
orientation(ms3ggt_0,152,387). coords(ms3ggt_0,142,140,387).  visible(ms3ggt_0,387). happensAt(active(ms3ggt_0),387). orientation(ms3ggt_1,162,387). coords(ms3ggt_1,142,157,387).  visible(ms3ggt_1,387). happensAt(walking(ms3ggt_1),387).
orientation(ms3ggt_0,152,388). coords(ms3ggt_0,142,140,388).  visible(ms3ggt_0,388). happensAt(active(ms3ggt_0),388). orientation(ms3ggt_1,162,388). coords(ms3ggt_1,143,157,388).  visible(ms3ggt_1,388). happensAt(walking(ms3ggt_1),388).
orientation(ms3ggt_0,152,389). coords(ms3ggt_0,142,139,389).  visible(ms3ggt_0,389). happensAt(active(ms3ggt_0),389). orientation(ms3ggt_1,162,389). coords(ms3ggt_1,143,157,389).  visible(ms3ggt_1,389). happensAt(walking(ms3ggt_1),389).
orientation(ms3ggt_0,152,390). coords(ms3ggt_0,142,139,390).  visible(ms3ggt_0,390). happensAt(active(ms3ggt_0),390). orientation(ms3ggt_1,162,390). coords(ms3ggt_1,143,157,390).  visible(ms3ggt_1,390). happensAt(walking(ms3ggt_1),390).
orientation(ms3ggt_0,152,391). coords(ms3ggt_0,142,138,391).  visible(ms3ggt_0,391). happensAt(active(ms3ggt_0),391). orientation(ms3ggt_1,162,391). coords(ms3ggt_1,144,157,391).  visible(ms3ggt_1,391). happensAt(walking(ms3ggt_1),391).
orientation(ms3ggt_0,152,392). coords(ms3ggt_0,142,138,392).  visible(ms3ggt_0,392). happensAt(active(ms3ggt_0),392). orientation(ms3ggt_1,162,392). coords(ms3ggt_1,144,158,392).  visible(ms3ggt_1,392). happensAt(walking(ms3ggt_1),392).
orientation(ms3ggt_0,152,393). coords(ms3ggt_0,142,138,393).  visible(ms3ggt_0,393). happensAt(active(ms3ggt_0),393). orientation(ms3ggt_1,153,393). coords(ms3ggt_1,144,158,393).  visible(ms3ggt_1,393). happensAt(walking(ms3ggt_1),393).
orientation(ms3ggt_0,152,394). coords(ms3ggt_0,141,138,394).  visible(ms3ggt_0,394). happensAt(active(ms3ggt_0),394). orientation(ms3ggt_1,153,394). coords(ms3ggt_1,147,158,394).  visible(ms3ggt_1,394). happensAt(walking(ms3ggt_1),394).
orientation(ms3ggt_0,152,395). coords(ms3ggt_0,141,138,395).  visible(ms3ggt_0,395). happensAt(walking(ms3ggt_0),395). orientation(ms3ggt_1,153,395). coords(ms3ggt_1,147,158,395).  visible(ms3ggt_1,395). happensAt(walking(ms3ggt_1),395).
orientation(ms3ggt_0,158,396). coords(ms3ggt_0,141,138,396).  visible(ms3ggt_0,396). happensAt(walking(ms3ggt_0),396). orientation(ms3ggt_1,153,396). coords(ms3ggt_1,149,158,396).  visible(ms3ggt_1,396). happensAt(walking(ms3ggt_1),396).
orientation(ms3ggt_0,158,397). coords(ms3ggt_0,140,138,397).  visible(ms3ggt_0,397). happensAt(walking(ms3ggt_0),397). orientation(ms3ggt_1,153,397). coords(ms3ggt_1,149,158,397).  visible(ms3ggt_1,397). happensAt(walking(ms3ggt_1),397).
orientation(ms3ggt_0,158,398). coords(ms3ggt_0,140,138,398).  visible(ms3ggt_0,398). happensAt(walking(ms3ggt_0),398). orientation(ms3ggt_1,153,398). coords(ms3ggt_1,149,158,398).  visible(ms3ggt_1,398). happensAt(walking(ms3ggt_1),398).
orientation(ms3ggt_0,158,399). coords(ms3ggt_0,139,138,399).  visible(ms3ggt_0,399). happensAt(walking(ms3ggt_0),399). orientation(ms3ggt_1,153,399). coords(ms3ggt_1,149,158,399).  visible(ms3ggt_1,399). happensAt(walking(ms3ggt_1),399).
orientation(ms3ggt_0,158,400). coords(ms3ggt_0,139,138,400).  visible(ms3ggt_0,400). happensAt(walking(ms3ggt_0),400). orientation(ms3ggt_1,153,400). coords(ms3ggt_1,150,158,400).  visible(ms3ggt_1,400). happensAt(walking(ms3ggt_1),400).
orientation(ms3ggt_0,161,401). coords(ms3ggt_0,139,138,401).  visible(ms3ggt_0,401). happensAt(walking(ms3ggt_0),401). orientation(ms3ggt_1,153,401). coords(ms3ggt_1,150,159,401).  visible(ms3ggt_1,401). happensAt(walking(ms3ggt_1),401).
orientation(ms3ggt_0,161,402). coords(ms3ggt_0,138,138,402).  visible(ms3ggt_0,402). happensAt(walking(ms3ggt_0),402). orientation(ms3ggt_1,153,402). coords(ms3ggt_1,150,159,402).  visible(ms3ggt_1,402). happensAt(walking(ms3ggt_1),402).
orientation(ms3ggt_0,161,403). coords(ms3ggt_0,137,138,403).  visible(ms3ggt_0,403). happensAt(walking(ms3ggt_0),403). orientation(ms3ggt_1,153,403). coords(ms3ggt_1,150,159,403).  visible(ms3ggt_1,403). happensAt(walking(ms3ggt_1),403).
orientation(ms3ggt_0,156,404). coords(ms3ggt_0,136,139,404).  visible(ms3ggt_0,404). happensAt(walking(ms3ggt_0),404). orientation(ms3ggt_1,153,404). coords(ms3ggt_1,151,159,404).  visible(ms3ggt_1,404). happensAt(walking(ms3ggt_1),404).
orientation(ms3ggt_0,156,405). coords(ms3ggt_0,134,139,405).  visible(ms3ggt_0,405). happensAt(walking(ms3ggt_0),405). orientation(ms3ggt_1,153,405). coords(ms3ggt_1,151,159,405).  visible(ms3ggt_1,405). happensAt(walking(ms3ggt_1),405).
orientation(ms3ggt_0,156,406). coords(ms3ggt_0,133,139,406).  visible(ms3ggt_0,406). happensAt(walking(ms3ggt_0),406). orientation(ms3ggt_1,153,406). coords(ms3ggt_1,152,159,406).  visible(ms3ggt_1,406). happensAt(walking(ms3ggt_1),406).
orientation(ms3ggt_0,156,407). coords(ms3ggt_0,132,140,407).  visible(ms3ggt_0,407). happensAt(walking(ms3ggt_0),407). orientation(ms3ggt_1,153,407). coords(ms3ggt_1,152,159,407).  visible(ms3ggt_1,407). happensAt(walking(ms3ggt_1),407).
orientation(ms3ggt_0,158,408). coords(ms3ggt_0,131,140,408).  visible(ms3ggt_0,408). happensAt(walking(ms3ggt_0),408). orientation(ms3ggt_1,153,408). coords(ms3ggt_1,152,159,408).  visible(ms3ggt_1,408). happensAt(walking(ms3ggt_1),408).
orientation(ms3ggt_0,158,409). coords(ms3ggt_0,131,140,409).  visible(ms3ggt_0,409). happensAt(walking(ms3ggt_0),409). orientation(ms3ggt_1,153,409). coords(ms3ggt_1,152,159,409).  visible(ms3ggt_1,409). happensAt(walking(ms3ggt_1),409).
orientation(ms3ggt_0,158,410). coords(ms3ggt_0,131,140,410).  visible(ms3ggt_0,410). happensAt(walking(ms3ggt_0),410). orientation(ms3ggt_1,153,410). coords(ms3ggt_1,153,158,410).  visible(ms3ggt_1,410). happensAt(walking(ms3ggt_1),410).
#show holdsAt/2.