
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

	orientation(ms3ggt_0,158,380). coords(ms3ggt_0,141,142,380).  visible(ms3ggt_0,380). happensAt(active(ms3ggt_0),380). orientation(ms3ggt_2,173,380). coords(ms3ggt_2,166,210,380).  visible(ms3ggt_2,380). happensAt(walking(ms3ggt_2),380).
orientation(ms3ggt_0,152,381). coords(ms3ggt_0,141,142,381).  visible(ms3ggt_0,381). happensAt(active(ms3ggt_0),381). orientation(ms3ggt_2,173,381). coords(ms3ggt_2,165,209,381).  visible(ms3ggt_2,381). happensAt(walking(ms3ggt_2),381).
orientation(ms3ggt_0,152,382). coords(ms3ggt_0,142,141,382).  visible(ms3ggt_0,382). happensAt(active(ms3ggt_0),382). orientation(ms3ggt_2,165,382). coords(ms3ggt_2,163,208,382).  visible(ms3ggt_2,382). happensAt(walking(ms3ggt_2),382).
orientation(ms3ggt_0,152,383). coords(ms3ggt_0,142,141,383).  visible(ms3ggt_0,383). happensAt(active(ms3ggt_0),383). orientation(ms3ggt_2,165,383). coords(ms3ggt_2,159,206,383).  visible(ms3ggt_2,383). happensAt(walking(ms3ggt_2),383).
orientation(ms3ggt_0,152,384). coords(ms3ggt_0,142,140,384).  visible(ms3ggt_0,384). happensAt(active(ms3ggt_0),384). orientation(ms3ggt_2,165,384). coords(ms3ggt_2,159,204,384).  visible(ms3ggt_2,384). happensAt(walking(ms3ggt_2),384).
orientation(ms3ggt_0,152,385). coords(ms3ggt_0,142,140,385).  visible(ms3ggt_0,385). happensAt(active(ms3ggt_0),385). orientation(ms3ggt_2,168,385). coords(ms3ggt_2,155,203,385).  visible(ms3ggt_2,385). happensAt(walking(ms3ggt_2),385).
orientation(ms3ggt_0,152,386). coords(ms3ggt_0,142,140,386).  visible(ms3ggt_0,386). happensAt(active(ms3ggt_0),386). orientation(ms3ggt_2,168,386). coords(ms3ggt_2,154,203,386).  visible(ms3ggt_2,386). happensAt(walking(ms3ggt_2),386).
orientation(ms3ggt_0,152,387). coords(ms3ggt_0,142,140,387).  visible(ms3ggt_0,387). happensAt(active(ms3ggt_0),387). orientation(ms3ggt_2,168,387). coords(ms3ggt_2,153,201,387).  visible(ms3ggt_2,387). happensAt(walking(ms3ggt_2),387).
orientation(ms3ggt_0,152,388). coords(ms3ggt_0,142,140,388).  visible(ms3ggt_0,388). happensAt(active(ms3ggt_0),388). orientation(ms3ggt_2,168,388). coords(ms3ggt_2,153,201,388).  visible(ms3ggt_2,388). happensAt(walking(ms3ggt_2),388).
orientation(ms3ggt_0,152,389). coords(ms3ggt_0,142,139,389).  visible(ms3ggt_0,389). happensAt(active(ms3ggt_0),389). orientation(ms3ggt_2,168,389). coords(ms3ggt_2,153,201,389).  visible(ms3ggt_2,389). happensAt(walking(ms3ggt_2),389).
orientation(ms3ggt_0,152,390). coords(ms3ggt_0,142,139,390).  visible(ms3ggt_0,390). happensAt(active(ms3ggt_0),390). orientation(ms3ggt_2,168,390). coords(ms3ggt_2,152,201,390).  visible(ms3ggt_2,390). happensAt(walking(ms3ggt_2),390).
orientation(ms3ggt_0,152,391). coords(ms3ggt_0,142,138,391).  visible(ms3ggt_0,391). happensAt(active(ms3ggt_0),391). orientation(ms3ggt_2,168,391). coords(ms3ggt_2,152,200,391).  visible(ms3ggt_2,391). happensAt(walking(ms3ggt_2),391).
orientation(ms3ggt_0,152,392). coords(ms3ggt_0,142,138,392).  visible(ms3ggt_0,392). happensAt(active(ms3ggt_0),392). orientation(ms3ggt_2,168,392). coords(ms3ggt_2,151,200,392).  visible(ms3ggt_2,392). happensAt(walking(ms3ggt_2),392).
orientation(ms3ggt_0,152,393). coords(ms3ggt_0,142,138,393).  visible(ms3ggt_0,393). happensAt(active(ms3ggt_0),393). orientation(ms3ggt_2,168,393). coords(ms3ggt_2,150,200,393).  visible(ms3ggt_2,393). happensAt(walking(ms3ggt_2),393).
orientation(ms3ggt_0,152,394). coords(ms3ggt_0,141,138,394).  visible(ms3ggt_0,394). happensAt(active(ms3ggt_0),394). orientation(ms3ggt_2,168,394). coords(ms3ggt_2,149,199,394).  visible(ms3ggt_2,394). happensAt(walking(ms3ggt_2),394).
orientation(ms3ggt_0,152,395). coords(ms3ggt_0,141,138,395).  visible(ms3ggt_0,395). happensAt(walking(ms3ggt_0),395). orientation(ms3ggt_2,168,395). coords(ms3ggt_2,148,199,395).  visible(ms3ggt_2,395). happensAt(walking(ms3ggt_2),395).
orientation(ms3ggt_0,158,396). coords(ms3ggt_0,141,138,396).  visible(ms3ggt_0,396). happensAt(walking(ms3ggt_0),396). orientation(ms3ggt_2,168,396). coords(ms3ggt_2,146,199,396).  visible(ms3ggt_2,396). happensAt(walking(ms3ggt_2),396).
orientation(ms3ggt_0,158,397). coords(ms3ggt_0,140,138,397).  visible(ms3ggt_0,397). happensAt(walking(ms3ggt_0),397). orientation(ms3ggt_2,168,397). coords(ms3ggt_2,145,199,397).  visible(ms3ggt_2,397). happensAt(walking(ms3ggt_2),397).
orientation(ms3ggt_0,158,398). coords(ms3ggt_0,140,138,398).  visible(ms3ggt_0,398). happensAt(walking(ms3ggt_0),398). orientation(ms3ggt_2,168,398). coords(ms3ggt_2,142,196,398).  visible(ms3ggt_2,398). happensAt(walking(ms3ggt_2),398).
orientation(ms3ggt_0,158,399). coords(ms3ggt_0,139,138,399).  visible(ms3ggt_0,399). happensAt(walking(ms3ggt_0),399). orientation(ms3ggt_2,168,399). coords(ms3ggt_2,141,196,399).  visible(ms3ggt_2,399). happensAt(walking(ms3ggt_2),399).
orientation(ms3ggt_0,158,400). coords(ms3ggt_0,139,138,400).  visible(ms3ggt_0,400). happensAt(walking(ms3ggt_0),400). orientation(ms3ggt_2,168,400). coords(ms3ggt_2,140,195,400).  visible(ms3ggt_2,400). happensAt(walking(ms3ggt_2),400).
orientation(ms3ggt_0,161,401). coords(ms3ggt_0,139,138,401).  visible(ms3ggt_0,401). happensAt(walking(ms3ggt_0),401). orientation(ms3ggt_2,168,401). coords(ms3ggt_2,139,194,401).  visible(ms3ggt_2,401). happensAt(walking(ms3ggt_2),401).
orientation(ms3ggt_0,161,402). coords(ms3ggt_0,138,138,402).  visible(ms3ggt_0,402). happensAt(walking(ms3ggt_0),402). orientation(ms3ggt_2,168,402). coords(ms3ggt_2,138,194,402).  visible(ms3ggt_2,402). happensAt(walking(ms3ggt_2),402).
orientation(ms3ggt_0,161,403). coords(ms3ggt_0,137,138,403).  visible(ms3ggt_0,403). happensAt(walking(ms3ggt_0),403). orientation(ms3ggt_2,168,403). coords(ms3ggt_2,137,193,403).  visible(ms3ggt_2,403). happensAt(walking(ms3ggt_2),403).
orientation(ms3ggt_0,156,404). coords(ms3ggt_0,136,139,404).  visible(ms3ggt_0,404). happensAt(walking(ms3ggt_0),404). orientation(ms3ggt_2,159,404). coords(ms3ggt_2,137,193,404).  visible(ms3ggt_2,404). happensAt(walking(ms3ggt_2),404).
#show holdsAt/2.