
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

	orientation(wk2gt_1,125,379). coords(wk2gt_1,115,53,379).  visible(wk2gt_1,379). happensAt(inactive(wk2gt_1),379). orientation(wk2gt_4,147,379). coords(wk2gt_4,156,128,379).  visible(wk2gt_4,379). happensAt(walking(wk2gt_4),379).
orientation(wk2gt_1,125,380). coords(wk2gt_1,115,53,380).  visible(wk2gt_1,380). happensAt(inactive(wk2gt_1),380). orientation(wk2gt_4,147,380). coords(wk2gt_4,155,126,380).  visible(wk2gt_4,380). happensAt(walking(wk2gt_4),380).
orientation(wk2gt_1,125,381). coords(wk2gt_1,115,53,381).  visible(wk2gt_1,381). happensAt(inactive(wk2gt_1),381). orientation(wk2gt_4,147,381). coords(wk2gt_4,153,125,381).  visible(wk2gt_4,381). happensAt(walking(wk2gt_4),381).
orientation(wk2gt_1,125,382). coords(wk2gt_1,115,53,382).  visible(wk2gt_1,382). happensAt(inactive(wk2gt_1),382). orientation(wk2gt_4,147,382). coords(wk2gt_4,153,124,382).  visible(wk2gt_4,382). happensAt(walking(wk2gt_4),382).
orientation(wk2gt_1,125,383). coords(wk2gt_1,115,53,383).  visible(wk2gt_1,383). happensAt(inactive(wk2gt_1),383). orientation(wk2gt_4,147,383). coords(wk2gt_4,152,122,383).  visible(wk2gt_4,383). happensAt(walking(wk2gt_4),383).
orientation(wk2gt_1,125,384). coords(wk2gt_1,115,53,384).  visible(wk2gt_1,384). happensAt(inactive(wk2gt_1),384). orientation(wk2gt_4,147,384). coords(wk2gt_4,152,122,384).  visible(wk2gt_4,384). happensAt(walking(wk2gt_4),384).
orientation(wk2gt_1,125,385). coords(wk2gt_1,115,53,385).  visible(wk2gt_1,385). happensAt(inactive(wk2gt_1),385). orientation(wk2gt_4,147,385). coords(wk2gt_4,151,121,385).  visible(wk2gt_4,385). happensAt(walking(wk2gt_4),385).
orientation(wk2gt_1,125,386). coords(wk2gt_1,115,53,386).  visible(wk2gt_1,386). happensAt(inactive(wk2gt_1),386). orientation(wk2gt_4,147,386). coords(wk2gt_4,150,121,386).  visible(wk2gt_4,386). happensAt(walking(wk2gt_4),386).
orientation(wk2gt_1,125,387). coords(wk2gt_1,115,53,387).  visible(wk2gt_1,387). happensAt(inactive(wk2gt_1),387). orientation(wk2gt_4,149,387). coords(wk2gt_4,150,121,387).  visible(wk2gt_4,387). happensAt(walking(wk2gt_4),387).
orientation(wk2gt_1,125,388). coords(wk2gt_1,115,53,388).  visible(wk2gt_1,388). happensAt(inactive(wk2gt_1),388). orientation(wk2gt_4,149,388). coords(wk2gt_4,149,120,388).  visible(wk2gt_4,388). happensAt(walking(wk2gt_4),388).
orientation(wk2gt_1,125,389). coords(wk2gt_1,115,53,389).  visible(wk2gt_1,389). happensAt(inactive(wk2gt_1),389). orientation(wk2gt_4,149,389). coords(wk2gt_4,148,119,389).  visible(wk2gt_4,389). happensAt(walking(wk2gt_4),389).
orientation(wk2gt_1,125,390). coords(wk2gt_1,115,53,390).  visible(wk2gt_1,390). happensAt(inactive(wk2gt_1),390). orientation(wk2gt_4,149,390). coords(wk2gt_4,146,118,390).  visible(wk2gt_4,390). happensAt(walking(wk2gt_4),390).
orientation(wk2gt_1,125,391). coords(wk2gt_1,115,53,391).  visible(wk2gt_1,391). happensAt(inactive(wk2gt_1),391). orientation(wk2gt_4,149,391). coords(wk2gt_4,144,117,391).  visible(wk2gt_4,391). happensAt(walking(wk2gt_4),391).
orientation(wk2gt_1,125,392). coords(wk2gt_1,115,53,392).  visible(wk2gt_1,392). happensAt(inactive(wk2gt_1),392). orientation(wk2gt_4,149,392). coords(wk2gt_4,142,117,392).  visible(wk2gt_4,392). happensAt(walking(wk2gt_4),392).
orientation(wk2gt_1,125,393). coords(wk2gt_1,115,53,393).  visible(wk2gt_1,393). happensAt(inactive(wk2gt_1),393). orientation(wk2gt_4,149,393). coords(wk2gt_4,139,116,393).  visible(wk2gt_4,393). happensAt(walking(wk2gt_4),393).
orientation(wk2gt_1,125,394). coords(wk2gt_1,115,53,394).  visible(wk2gt_1,394). happensAt(inactive(wk2gt_1),394). orientation(wk2gt_4,149,394). coords(wk2gt_4,138,116,394).  visible(wk2gt_4,394). happensAt(walking(wk2gt_4),394).
orientation(wk2gt_1,125,395). coords(wk2gt_1,115,53,395).  visible(wk2gt_1,395). happensAt(inactive(wk2gt_1),395). orientation(wk2gt_4,149,395). coords(wk2gt_4,137,115,395).  visible(wk2gt_4,395). happensAt(walking(wk2gt_4),395).
orientation(wk2gt_1,125,396). coords(wk2gt_1,115,53,396).  visible(wk2gt_1,396). happensAt(inactive(wk2gt_1),396). orientation(wk2gt_4,149,396). coords(wk2gt_4,136,115,396).  visible(wk2gt_4,396). happensAt(walking(wk2gt_4),396).
orientation(wk2gt_1,125,397). coords(wk2gt_1,115,53,397).  visible(wk2gt_1,397). happensAt(inactive(wk2gt_1),397). orientation(wk2gt_4,149,397). coords(wk2gt_4,136,115,397).  visible(wk2gt_4,397). happensAt(walking(wk2gt_4),397).
orientation(wk2gt_1,125,398). coords(wk2gt_1,115,53,398).  visible(wk2gt_1,398). happensAt(inactive(wk2gt_1),398). orientation(wk2gt_4,149,398). coords(wk2gt_4,136,115,398).  visible(wk2gt_4,398). happensAt(walking(wk2gt_4),398).
orientation(wk2gt_1,125,399). coords(wk2gt_1,115,53,399).  visible(wk2gt_1,399). happensAt(inactive(wk2gt_1),399). orientation(wk2gt_4,149,399). coords(wk2gt_4,135,114,399).  visible(wk2gt_4,399). happensAt(walking(wk2gt_4),399).
orientation(wk2gt_1,125,400). coords(wk2gt_1,115,53,400).  visible(wk2gt_1,400). happensAt(inactive(wk2gt_1),400). orientation(wk2gt_4,149,400). coords(wk2gt_4,133,114,400).  visible(wk2gt_4,400). happensAt(walking(wk2gt_4),400).
orientation(wk2gt_1,125,401). coords(wk2gt_1,115,53,401).  visible(wk2gt_1,401). happensAt(inactive(wk2gt_1),401). orientation(wk2gt_4,149,401). coords(wk2gt_4,133,114,401).  visible(wk2gt_4,401). happensAt(walking(wk2gt_4),401).
orientation(wk2gt_1,125,402). coords(wk2gt_1,115,53,402).  visible(wk2gt_1,402). happensAt(inactive(wk2gt_1),402). orientation(wk2gt_4,149,402). coords(wk2gt_4,130,112,402).  visible(wk2gt_4,402). happensAt(walking(wk2gt_4),402).
orientation(wk2gt_1,125,403). coords(wk2gt_1,115,53,403).  visible(wk2gt_1,403). happensAt(inactive(wk2gt_1),403). orientation(wk2gt_4,146,403). coords(wk2gt_4,128,112,403).  visible(wk2gt_4,403). happensAt(walking(wk2gt_4),403).
#show holdsAt/2.