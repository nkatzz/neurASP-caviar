
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

	orientation(fra2gt_3,138,382). coords(fra2gt_3,74,129,382).  visible(fra2gt_3,382). happensAt(walking(fra2gt_3),382). orientation(fra2gt_4,149,382). coords(fra2gt_4,78,132,382).  visible(fra2gt_4,382). happensAt(walking(fra2gt_4),382).
orientation(fra2gt_3,138,383). coords(fra2gt_3,73,132,383).  visible(fra2gt_3,383). happensAt(walking(fra2gt_3),383). orientation(fra2gt_4,145,383). coords(fra2gt_4,78,131,383).  visible(fra2gt_4,383). happensAt(walking(fra2gt_4),383).
orientation(fra2gt_3,151,384). coords(fra2gt_3,71,133,384).  visible(fra2gt_3,384). happensAt(walking(fra2gt_3),384). orientation(fra2gt_4,146,384). coords(fra2gt_4,77,133,384).  visible(fra2gt_4,384). happensAt(walking(fra2gt_4),384).
orientation(fra2gt_3,159,385). coords(fra2gt_3,71,133,385).  visible(fra2gt_3,385). happensAt(walking(fra2gt_3),385). orientation(fra2gt_4,143,385). coords(fra2gt_4,77,135,385).  visible(fra2gt_4,385). happensAt(walking(fra2gt_4),385).
orientation(fra2gt_3,159,386). coords(fra2gt_3,70,133,386).  visible(fra2gt_3,386). happensAt(walking(fra2gt_3),386). orientation(fra2gt_4,142,386). coords(fra2gt_4,77,136,386).  visible(fra2gt_4,386). happensAt(walking(fra2gt_4),386).
orientation(fra2gt_3,166,387). coords(fra2gt_3,70,133,387).  visible(fra2gt_3,387). happensAt(walking(fra2gt_3),387). orientation(fra2gt_4,142,387). coords(fra2gt_4,77,137,387).  visible(fra2gt_4,387). happensAt(walking(fra2gt_4),387).
orientation(fra2gt_3,161,388). coords(fra2gt_3,69,135,388).  visible(fra2gt_3,388). happensAt(walking(fra2gt_3),388). orientation(fra2gt_4,142,388). coords(fra2gt_4,77,138,388).  visible(fra2gt_4,388). happensAt(walking(fra2gt_4),388).
orientation(fra2gt_3,161,389). coords(fra2gt_3,69,136,389).  visible(fra2gt_3,389). happensAt(walking(fra2gt_3),389). orientation(fra2gt_4,142,389). coords(fra2gt_4,77,138,389).  visible(fra2gt_4,389). happensAt(walking(fra2gt_4),389).
orientation(fra2gt_3,158,390). coords(fra2gt_3,69,137,390).  visible(fra2gt_3,390). happensAt(walking(fra2gt_3),390). orientation(fra2gt_4,142,390). coords(fra2gt_4,76,138,390).  visible(fra2gt_4,390). happensAt(walking(fra2gt_4),390).
orientation(fra2gt_3,159,391). coords(fra2gt_3,66,137,391).  visible(fra2gt_3,391). happensAt(walking(fra2gt_3),391). orientation(fra2gt_4,148,391). coords(fra2gt_4,76,138,391).  visible(fra2gt_4,391). happensAt(walking(fra2gt_4),391).
orientation(fra2gt_3,157,392). coords(fra2gt_3,66,138,392).  visible(fra2gt_3,392). happensAt(walking(fra2gt_3),392). orientation(fra2gt_4,148,392). coords(fra2gt_4,76,138,392).  visible(fra2gt_4,392). happensAt(walking(fra2gt_4),392).
orientation(fra2gt_3,167,393). coords(fra2gt_3,65,138,393).  visible(fra2gt_3,393). happensAt(walking(fra2gt_3),393). orientation(fra2gt_4,148,393). coords(fra2gt_4,77,138,393).  visible(fra2gt_4,393). happensAt(walking(fra2gt_4),393).
orientation(fra2gt_3,162,394). coords(fra2gt_3,65,138,394).  visible(fra2gt_3,394). happensAt(walking(fra2gt_3),394). orientation(fra2gt_4,148,394). coords(fra2gt_4,77,138,394).  visible(fra2gt_4,394). happensAt(walking(fra2gt_4),394).
orientation(fra2gt_3,162,395). coords(fra2gt_3,65,140,395).  visible(fra2gt_3,395). happensAt(walking(fra2gt_3),395). orientation(fra2gt_4,148,395). coords(fra2gt_4,77,140,395).  visible(fra2gt_4,395). happensAt(walking(fra2gt_4),395).
orientation(fra2gt_3,164,396). coords(fra2gt_3,65,140,396).  visible(fra2gt_3,396). happensAt(walking(fra2gt_3),396). orientation(fra2gt_4,148,396). coords(fra2gt_4,77,140,396).  visible(fra2gt_4,396). happensAt(walking(fra2gt_4),396).
orientation(fra2gt_3,164,397). coords(fra2gt_3,65,141,397).  visible(fra2gt_3,397). happensAt(walking(fra2gt_3),397). orientation(fra2gt_4,143,397). coords(fra2gt_4,76,140,397).  visible(fra2gt_4,397). happensAt(walking(fra2gt_4),397).
orientation(fra2gt_3,164,398). coords(fra2gt_3,65,141,398).  visible(fra2gt_3,398). happensAt(walking(fra2gt_3),398). orientation(fra2gt_4,142,398). coords(fra2gt_4,75,140,398).  visible(fra2gt_4,398). happensAt(walking(fra2gt_4),398).
orientation(fra2gt_3,164,399). coords(fra2gt_3,65,141,399).  visible(fra2gt_3,399). happensAt(walking(fra2gt_3),399). orientation(fra2gt_4,138,399). coords(fra2gt_4,75,140,399).  visible(fra2gt_4,399). happensAt(walking(fra2gt_4),399).
orientation(fra2gt_3,164,400). coords(fra2gt_3,65,142,400).  visible(fra2gt_3,400). happensAt(walking(fra2gt_3),400). orientation(fra2gt_4,138,400). coords(fra2gt_4,75,140,400).  visible(fra2gt_4,400). happensAt(walking(fra2gt_4),400).
orientation(fra2gt_3,164,401). coords(fra2gt_3,65,143,401).  visible(fra2gt_3,401). happensAt(walking(fra2gt_3),401). orientation(fra2gt_4,138,401). coords(fra2gt_4,75,139,401).  visible(fra2gt_4,401). happensAt(walking(fra2gt_4),401).
orientation(fra2gt_3,164,402). coords(fra2gt_3,64,143,402).  visible(fra2gt_3,402). happensAt(walking(fra2gt_3),402). orientation(fra2gt_4,138,402). coords(fra2gt_4,75,139,402).  visible(fra2gt_4,402). happensAt(walking(fra2gt_4),402).
orientation(fra2gt_3,164,403). coords(fra2gt_3,64,145,403).  visible(fra2gt_3,403). happensAt(walking(fra2gt_3),403). orientation(fra2gt_4,138,403). coords(fra2gt_4,75,137,403).  visible(fra2gt_4,403). happensAt(walking(fra2gt_4),403).
orientation(fra2gt_3,164,404). coords(fra2gt_3,62,145,404).  visible(fra2gt_3,404). happensAt(walking(fra2gt_3),404). orientation(fra2gt_4,138,404). coords(fra2gt_4,75,134,404).  visible(fra2gt_4,404). happensAt(running(fra2gt_4),404).
orientation(fra2gt_3,164,405). coords(fra2gt_3,62,145,405).  visible(fra2gt_3,405). happensAt(walking(fra2gt_3),405). orientation(fra2gt_4,138,405). coords(fra2gt_4,75,133,405).  visible(fra2gt_4,405). happensAt(running(fra2gt_4),405).
orientation(fra2gt_3,164,406). coords(fra2gt_3,62,145,406).  visible(fra2gt_3,406). happensAt(walking(fra2gt_3),406). orientation(fra2gt_4,138,406). coords(fra2gt_4,75,131,406).  visible(fra2gt_4,406). happensAt(running(fra2gt_4),406).
#show holdsAt/2.