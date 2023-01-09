
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

	orientation(br1gt_2,169,382). coords(br1gt_2,86,208,382).  visible(br1gt_2,382). happensAt(inactive(br1gt_2),382). orientation(br1gt_3,162,382). coords(br1gt_3,37,159,382).  visible(br1gt_3,382). happensAt(walking(br1gt_3),382).
orientation(br1gt_2,169,383). coords(br1gt_2,86,208,383).  visible(br1gt_2,383). happensAt(inactive(br1gt_2),383). orientation(br1gt_3,162,383). coords(br1gt_3,37,159,383).  visible(br1gt_3,383). happensAt(walking(br1gt_3),383).
orientation(br1gt_2,174,384). coords(br1gt_2,86,208,384).  visible(br1gt_2,384). happensAt(inactive(br1gt_2),384). orientation(br1gt_3,162,384). coords(br1gt_3,37,158,384).  visible(br1gt_3,384). happensAt(walking(br1gt_3),384).
orientation(br1gt_2,174,385). coords(br1gt_2,86,208,385).  visible(br1gt_2,385). happensAt(inactive(br1gt_2),385). orientation(br1gt_3,162,385). coords(br1gt_3,38,158,385).  visible(br1gt_3,385). happensAt(walking(br1gt_3),385).
orientation(br1gt_2,174,386). coords(br1gt_2,86,208,386).  visible(br1gt_2,386). happensAt(inactive(br1gt_2),386). orientation(br1gt_3,162,386). coords(br1gt_3,39,158,386).  visible(br1gt_3,386). happensAt(walking(br1gt_3),386).
orientation(br1gt_2,174,387). coords(br1gt_2,86,208,387).  visible(br1gt_2,387). happensAt(inactive(br1gt_2),387). orientation(br1gt_3,162,387). coords(br1gt_3,39,158,387).  visible(br1gt_3,387). happensAt(walking(br1gt_3),387).
orientation(br1gt_2,174,388). coords(br1gt_2,86,208,388).  visible(br1gt_2,388). happensAt(inactive(br1gt_2),388). orientation(br1gt_3,162,388). coords(br1gt_3,41,158,388).  visible(br1gt_3,388). happensAt(walking(br1gt_3),388).
orientation(br1gt_2,174,389). coords(br1gt_2,86,208,389).  visible(br1gt_2,389). happensAt(inactive(br1gt_2),389). orientation(br1gt_3,162,389). coords(br1gt_3,41,158,389).  visible(br1gt_3,389). happensAt(walking(br1gt_3),389).
orientation(br1gt_2,174,390). coords(br1gt_2,86,208,390).  visible(br1gt_2,390). happensAt(inactive(br1gt_2),390). orientation(br1gt_3,162,390). coords(br1gt_3,41,158,390).  visible(br1gt_3,390). happensAt(walking(br1gt_3),390).
orientation(br1gt_2,174,391). coords(br1gt_2,86,208,391).  visible(br1gt_2,391). happensAt(inactive(br1gt_2),391). orientation(br1gt_3,162,391). coords(br1gt_3,41,157,391).  visible(br1gt_3,391). happensAt(walking(br1gt_3),391).
orientation(br1gt_2,7,392). coords(br1gt_2,86,208,392).  visible(br1gt_2,392). happensAt(inactive(br1gt_2),392). orientation(br1gt_3,162,392). coords(br1gt_3,41,157,392).  visible(br1gt_3,392). happensAt(walking(br1gt_3),392).
orientation(br1gt_2,7,393). coords(br1gt_2,86,208,393).  visible(br1gt_2,393). happensAt(inactive(br1gt_2),393). orientation(br1gt_3,162,393). coords(br1gt_3,41,157,393).  visible(br1gt_3,393). happensAt(walking(br1gt_3),393).
orientation(br1gt_2,7,394). coords(br1gt_2,86,208,394).  visible(br1gt_2,394). happensAt(inactive(br1gt_2),394). orientation(br1gt_3,162,394). coords(br1gt_3,41,157,394).  visible(br1gt_3,394). happensAt(walking(br1gt_3),394).
orientation(br1gt_2,7,395). coords(br1gt_2,86,207,395).  visible(br1gt_2,395). happensAt(inactive(br1gt_2),395). orientation(br1gt_3,162,395). coords(br1gt_3,41,156,395).  visible(br1gt_3,395). happensAt(walking(br1gt_3),395).
orientation(br1gt_2,7,396). coords(br1gt_2,86,207,396).  visible(br1gt_2,396). happensAt(inactive(br1gt_2),396). orientation(br1gt_3,162,396). coords(br1gt_3,41,157,396).  visible(br1gt_3,396). happensAt(walking(br1gt_3),396).
orientation(br1gt_2,7,397). coords(br1gt_2,84,207,397).  visible(br1gt_2,397). happensAt(inactive(br1gt_2),397). orientation(br1gt_3,162,397). coords(br1gt_3,41,157,397).  visible(br1gt_3,397). happensAt(walking(br1gt_3),397).
orientation(br1gt_2,7,398). coords(br1gt_2,84,207,398).  visible(br1gt_2,398). happensAt(inactive(br1gt_2),398). orientation(br1gt_3,162,398). coords(br1gt_3,42,156,398).  visible(br1gt_3,398). happensAt(walking(br1gt_3),398).
orientation(br1gt_2,7,399). coords(br1gt_2,84,207,399).  visible(br1gt_2,399). happensAt(inactive(br1gt_2),399). orientation(br1gt_3,162,399). coords(br1gt_3,43,156,399).  visible(br1gt_3,399). happensAt(walking(br1gt_3),399).
orientation(br1gt_2,7,400). coords(br1gt_2,84,207,400).  visible(br1gt_2,400). happensAt(inactive(br1gt_2),400). orientation(br1gt_3,162,400). coords(br1gt_3,43,156,400).  visible(br1gt_3,400). happensAt(walking(br1gt_3),400).
orientation(br1gt_2,7,401). coords(br1gt_2,84,207,401).  visible(br1gt_2,401). happensAt(inactive(br1gt_2),401). orientation(br1gt_3,162,401). coords(br1gt_3,43,156,401).  visible(br1gt_3,401). happensAt(walking(br1gt_3),401).
orientation(br1gt_2,7,402). coords(br1gt_2,84,208,402).  visible(br1gt_2,402). happensAt(inactive(br1gt_2),402). orientation(br1gt_3,162,402). coords(br1gt_3,43,156,402).  visible(br1gt_3,402). happensAt(walking(br1gt_3),402).
orientation(br1gt_2,7,403). coords(br1gt_2,84,208,403).  visible(br1gt_2,403). happensAt(inactive(br1gt_2),403). orientation(br1gt_3,162,403). coords(br1gt_3,44,156,403).  visible(br1gt_3,403). happensAt(walking(br1gt_3),403).
orientation(br1gt_2,7,404). coords(br1gt_2,84,208,404).  visible(br1gt_2,404). happensAt(inactive(br1gt_2),404). orientation(br1gt_3,162,404). coords(br1gt_3,44,156,404).  visible(br1gt_3,404). happensAt(walking(br1gt_3),404).
orientation(br1gt_2,7,405). coords(br1gt_2,84,208,405).  visible(br1gt_2,405). happensAt(inactive(br1gt_2),405). orientation(br1gt_3,162,405). coords(br1gt_3,44,156,405).  visible(br1gt_3,405). happensAt(walking(br1gt_3),405).
orientation(br1gt_2,7,406). coords(br1gt_2,84,208,406).  visible(br1gt_2,406). happensAt(inactive(br1gt_2),406). orientation(br1gt_3,162,406). coords(br1gt_3,44,156,406).  visible(br1gt_3,406). happensAt(walking(br1gt_3),406).
#show holdsAt/2.