
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

	orientation(ms3ggt_1,162,380). coords(ms3ggt_1,138,158,380).  visible(ms3ggt_1,380). happensAt(walking(ms3ggt_1),380). orientation(ms3ggt_2,173,380). coords(ms3ggt_2,166,210,380).  visible(ms3ggt_2,380). happensAt(walking(ms3ggt_2),380).
orientation(ms3ggt_1,162,381). coords(ms3ggt_1,138,158,381).  visible(ms3ggt_1,381). happensAt(walking(ms3ggt_1),381). orientation(ms3ggt_2,173,381). coords(ms3ggt_2,165,209,381).  visible(ms3ggt_2,381). happensAt(walking(ms3ggt_2),381).
orientation(ms3ggt_1,162,382). coords(ms3ggt_1,140,158,382).  visible(ms3ggt_1,382). happensAt(walking(ms3ggt_1),382). orientation(ms3ggt_2,165,382). coords(ms3ggt_2,163,208,382).  visible(ms3ggt_2,382). happensAt(walking(ms3ggt_2),382).
orientation(ms3ggt_1,162,383). coords(ms3ggt_1,141,158,383).  visible(ms3ggt_1,383). happensAt(walking(ms3ggt_1),383). orientation(ms3ggt_2,165,383). coords(ms3ggt_2,159,206,383).  visible(ms3ggt_2,383). happensAt(walking(ms3ggt_2),383).
orientation(ms3ggt_1,162,384). coords(ms3ggt_1,142,158,384).  visible(ms3ggt_1,384). happensAt(walking(ms3ggt_1),384). orientation(ms3ggt_2,165,384). coords(ms3ggt_2,159,204,384).  visible(ms3ggt_2,384). happensAt(walking(ms3ggt_2),384).
orientation(ms3ggt_1,162,385). coords(ms3ggt_1,142,157,385).  visible(ms3ggt_1,385). happensAt(walking(ms3ggt_1),385). orientation(ms3ggt_2,168,385). coords(ms3ggt_2,155,203,385).  visible(ms3ggt_2,385). happensAt(walking(ms3ggt_2),385).
orientation(ms3ggt_1,162,386). coords(ms3ggt_1,142,157,386).  visible(ms3ggt_1,386). happensAt(walking(ms3ggt_1),386). orientation(ms3ggt_2,168,386). coords(ms3ggt_2,154,203,386).  visible(ms3ggt_2,386). happensAt(walking(ms3ggt_2),386).
orientation(ms3ggt_1,162,387). coords(ms3ggt_1,142,157,387).  visible(ms3ggt_1,387). happensAt(walking(ms3ggt_1),387). orientation(ms3ggt_2,168,387). coords(ms3ggt_2,153,201,387).  visible(ms3ggt_2,387). happensAt(walking(ms3ggt_2),387).
orientation(ms3ggt_1,162,388). coords(ms3ggt_1,143,157,388).  visible(ms3ggt_1,388). happensAt(walking(ms3ggt_1),388). orientation(ms3ggt_2,168,388). coords(ms3ggt_2,153,201,388).  visible(ms3ggt_2,388). happensAt(walking(ms3ggt_2),388).
orientation(ms3ggt_1,162,389). coords(ms3ggt_1,143,157,389).  visible(ms3ggt_1,389). happensAt(walking(ms3ggt_1),389). orientation(ms3ggt_2,168,389). coords(ms3ggt_2,153,201,389).  visible(ms3ggt_2,389). happensAt(walking(ms3ggt_2),389).
orientation(ms3ggt_1,162,390). coords(ms3ggt_1,143,157,390).  visible(ms3ggt_1,390). happensAt(walking(ms3ggt_1),390). orientation(ms3ggt_2,168,390). coords(ms3ggt_2,152,201,390).  visible(ms3ggt_2,390). happensAt(walking(ms3ggt_2),390).
orientation(ms3ggt_1,162,391). coords(ms3ggt_1,144,157,391).  visible(ms3ggt_1,391). happensAt(walking(ms3ggt_1),391). orientation(ms3ggt_2,168,391). coords(ms3ggt_2,152,200,391).  visible(ms3ggt_2,391). happensAt(walking(ms3ggt_2),391).
orientation(ms3ggt_1,162,392). coords(ms3ggt_1,144,158,392).  visible(ms3ggt_1,392). happensAt(walking(ms3ggt_1),392). orientation(ms3ggt_2,168,392). coords(ms3ggt_2,151,200,392).  visible(ms3ggt_2,392). happensAt(walking(ms3ggt_2),392).
orientation(ms3ggt_1,153,393). coords(ms3ggt_1,144,158,393).  visible(ms3ggt_1,393). happensAt(walking(ms3ggt_1),393). orientation(ms3ggt_2,168,393). coords(ms3ggt_2,150,200,393).  visible(ms3ggt_2,393). happensAt(walking(ms3ggt_2),393).
orientation(ms3ggt_1,153,394). coords(ms3ggt_1,147,158,394).  visible(ms3ggt_1,394). happensAt(walking(ms3ggt_1),394). orientation(ms3ggt_2,168,394). coords(ms3ggt_2,149,199,394).  visible(ms3ggt_2,394). happensAt(walking(ms3ggt_2),394).
orientation(ms3ggt_1,153,395). coords(ms3ggt_1,147,158,395).  visible(ms3ggt_1,395). happensAt(walking(ms3ggt_1),395). orientation(ms3ggt_2,168,395). coords(ms3ggt_2,148,199,395).  visible(ms3ggt_2,395). happensAt(walking(ms3ggt_2),395).
orientation(ms3ggt_1,153,396). coords(ms3ggt_1,149,158,396).  visible(ms3ggt_1,396). happensAt(walking(ms3ggt_1),396). orientation(ms3ggt_2,168,396). coords(ms3ggt_2,146,199,396).  visible(ms3ggt_2,396). happensAt(walking(ms3ggt_2),396).
orientation(ms3ggt_1,153,397). coords(ms3ggt_1,149,158,397).  visible(ms3ggt_1,397). happensAt(walking(ms3ggt_1),397). orientation(ms3ggt_2,168,397). coords(ms3ggt_2,145,199,397).  visible(ms3ggt_2,397). happensAt(walking(ms3ggt_2),397).
orientation(ms3ggt_1,153,398). coords(ms3ggt_1,149,158,398).  visible(ms3ggt_1,398). happensAt(walking(ms3ggt_1),398). orientation(ms3ggt_2,168,398). coords(ms3ggt_2,142,196,398).  visible(ms3ggt_2,398). happensAt(walking(ms3ggt_2),398).
orientation(ms3ggt_1,153,399). coords(ms3ggt_1,149,158,399).  visible(ms3ggt_1,399). happensAt(walking(ms3ggt_1),399). orientation(ms3ggt_2,168,399). coords(ms3ggt_2,141,196,399).  visible(ms3ggt_2,399). happensAt(walking(ms3ggt_2),399).
orientation(ms3ggt_1,153,400). coords(ms3ggt_1,150,158,400).  visible(ms3ggt_1,400). happensAt(walking(ms3ggt_1),400). orientation(ms3ggt_2,168,400). coords(ms3ggt_2,140,195,400).  visible(ms3ggt_2,400). happensAt(walking(ms3ggt_2),400).
orientation(ms3ggt_1,153,401). coords(ms3ggt_1,150,159,401).  visible(ms3ggt_1,401). happensAt(walking(ms3ggt_1),401). orientation(ms3ggt_2,168,401). coords(ms3ggt_2,139,194,401).  visible(ms3ggt_2,401). happensAt(walking(ms3ggt_2),401).
orientation(ms3ggt_1,153,402). coords(ms3ggt_1,150,159,402).  visible(ms3ggt_1,402). happensAt(walking(ms3ggt_1),402). orientation(ms3ggt_2,168,402). coords(ms3ggt_2,138,194,402).  visible(ms3ggt_2,402). happensAt(walking(ms3ggt_2),402).
orientation(ms3ggt_1,153,403). coords(ms3ggt_1,150,159,403).  visible(ms3ggt_1,403). happensAt(walking(ms3ggt_1),403). orientation(ms3ggt_2,168,403). coords(ms3ggt_2,137,193,403).  visible(ms3ggt_2,403). happensAt(walking(ms3ggt_2),403).
orientation(ms3ggt_1,153,404). coords(ms3ggt_1,151,159,404).  visible(ms3ggt_1,404). happensAt(walking(ms3ggt_1),404). orientation(ms3ggt_2,159,404). coords(ms3ggt_2,137,193,404).  visible(ms3ggt_2,404). happensAt(walking(ms3ggt_2),404).
#show holdsAt/2.