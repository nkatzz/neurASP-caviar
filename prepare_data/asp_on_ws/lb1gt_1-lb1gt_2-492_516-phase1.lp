
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

	orientation(lb1gt_1,29,492). coords(lb1gt_1,212,268,492).  visible(lb1gt_1,492). happensAt(walking(lb1gt_1),492). orientation(lb1gt_2,1,492). coords(lb1gt_2,199,208,492).  visible(lb1gt_2,492). happensAt(walking(lb1gt_2),492).
orientation(lb1gt_1,31,493). coords(lb1gt_1,212,268,493).  visible(lb1gt_1,493). happensAt(walking(lb1gt_1),493). orientation(lb1gt_2,1,493). coords(lb1gt_2,199,209,493).  visible(lb1gt_2,493). happensAt(walking(lb1gt_2),493).
orientation(lb1gt_1,25,494). coords(lb1gt_1,212,269,494).  visible(lb1gt_1,494). happensAt(walking(lb1gt_1),494). orientation(lb1gt_2,1,494). coords(lb1gt_2,199,212,494).  visible(lb1gt_2,494). happensAt(walking(lb1gt_2),494).
orientation(lb1gt_1,25,495). coords(lb1gt_1,212,271,495).  visible(lb1gt_1,495). happensAt(walking(lb1gt_1),495). orientation(lb1gt_2,1,495). coords(lb1gt_2,199,214,495).  visible(lb1gt_2,495). happensAt(walking(lb1gt_2),495).
orientation(lb1gt_1,22,496). coords(lb1gt_1,212,272,496).  visible(lb1gt_1,496). happensAt(walking(lb1gt_1),496). orientation(lb1gt_2,1,496). coords(lb1gt_2,199,216,496).  visible(lb1gt_2,496). happensAt(walking(lb1gt_2),496).
orientation(lb1gt_1,22,497). coords(lb1gt_1,213,273,497).  visible(lb1gt_1,497). happensAt(walking(lb1gt_1),497). orientation(lb1gt_2,1,497). coords(lb1gt_2,199,221,497).  visible(lb1gt_2,497). happensAt(walking(lb1gt_2),497).
orientation(lb1gt_1,22,498). coords(lb1gt_1,213,275,498).  visible(lb1gt_1,498). happensAt(walking(lb1gt_1),498). orientation(lb1gt_2,1,498). coords(lb1gt_2,200,222,498).  visible(lb1gt_2,498). happensAt(walking(lb1gt_2),498).
orientation(lb1gt_1,22,499). coords(lb1gt_1,215,276,499).  visible(lb1gt_1,499). happensAt(walking(lb1gt_1),499). orientation(lb1gt_2,1,499). coords(lb1gt_2,200,224,499).  visible(lb1gt_2,499). happensAt(walking(lb1gt_2),499).
orientation(lb1gt_1,22,500). coords(lb1gt_1,217,278,500).  visible(lb1gt_1,500). happensAt(walking(lb1gt_1),500). orientation(lb1gt_2,1,500). coords(lb1gt_2,200,224,500).  visible(lb1gt_2,500). happensAt(walking(lb1gt_2),500).
orientation(lb1gt_1,16,501). coords(lb1gt_1,220,280,501).  visible(lb1gt_1,501). happensAt(walking(lb1gt_1),501). orientation(lb1gt_2,1,501). coords(lb1gt_2,200,224,501).  visible(lb1gt_2,501). happensAt(walking(lb1gt_2),501).
orientation(lb1gt_1,16,502). coords(lb1gt_1,221,281,502).  visible(lb1gt_1,502). happensAt(walking(lb1gt_1),502). orientation(lb1gt_2,1,502). coords(lb1gt_2,200,225,502).  visible(lb1gt_2,502). happensAt(walking(lb1gt_2),502).
orientation(lb1gt_1,16,503). coords(lb1gt_1,223,281,503).  visible(lb1gt_1,503). happensAt(walking(lb1gt_1),503). orientation(lb1gt_2,1,503). coords(lb1gt_2,200,225,503).  visible(lb1gt_2,503). happensAt(walking(lb1gt_2),503).
orientation(lb1gt_1,16,504). coords(lb1gt_1,224,281,504).  visible(lb1gt_1,504). happensAt(walking(lb1gt_1),504). orientation(lb1gt_2,6,504). coords(lb1gt_2,200,226,504).  visible(lb1gt_2,504). happensAt(walking(lb1gt_2),504).
orientation(lb1gt_1,16,505). coords(lb1gt_1,224,281,505).  visible(lb1gt_1,505). happensAt(walking(lb1gt_1),505). orientation(lb1gt_2,6,505). coords(lb1gt_2,201,228,505).  visible(lb1gt_2,505). happensAt(walking(lb1gt_2),505).
orientation(lb1gt_1,53,506). coords(lb1gt_1,225,281,506).  visible(lb1gt_1,506). happensAt(walking(lb1gt_1),506). orientation(lb1gt_2,6,506). coords(lb1gt_2,201,229,506).  visible(lb1gt_2,506). happensAt(walking(lb1gt_2),506).
orientation(lb1gt_1,53,507). coords(lb1gt_1,226,281,507).  visible(lb1gt_1,507). happensAt(walking(lb1gt_1),507). orientation(lb1gt_2,6,507). coords(lb1gt_2,201,231,507).  visible(lb1gt_2,507). happensAt(walking(lb1gt_2),507).
orientation(lb1gt_1,53,508). coords(lb1gt_1,226,281,508).  visible(lb1gt_1,508). happensAt(walking(lb1gt_1),508). orientation(lb1gt_2,6,508). coords(lb1gt_2,202,235,508).  visible(lb1gt_2,508). happensAt(walking(lb1gt_2),508).
orientation(lb1gt_1,53,509). coords(lb1gt_1,226,282,509).  visible(lb1gt_1,509). happensAt(walking(lb1gt_1),509). orientation(lb1gt_2,6,509). coords(lb1gt_2,203,237,509).  visible(lb1gt_2,509). happensAt(walking(lb1gt_2),509).
orientation(lb1gt_1,65,510). coords(lb1gt_1,232,282,510).  visible(lb1gt_1,510). happensAt(walking(lb1gt_1),510). orientation(lb1gt_2,6,510). coords(lb1gt_2,204,240,510).  visible(lb1gt_2,510). happensAt(walking(lb1gt_2),510).
orientation(lb1gt_1,77,511). coords(lb1gt_1,232,283,511).  visible(lb1gt_1,511). happensAt(walking(lb1gt_1),511). orientation(lb1gt_2,6,511). coords(lb1gt_2,206,242,511).  visible(lb1gt_2,511). happensAt(walking(lb1gt_2),511).
orientation(lb1gt_1,81,512). coords(lb1gt_1,233,283,512).  visible(lb1gt_1,512). happensAt(walking(lb1gt_1),512). orientation(lb1gt_2,6,512). coords(lb1gt_2,206,244,512).  visible(lb1gt_2,512). happensAt(walking(lb1gt_2),512).
orientation(lb1gt_1,81,513). coords(lb1gt_1,235,283,513).  visible(lb1gt_1,513). happensAt(walking(lb1gt_1),513). orientation(lb1gt_2,6,513). coords(lb1gt_2,207,245,513).  visible(lb1gt_2,513). happensAt(walking(lb1gt_2),513).
orientation(lb1gt_1,81,514). coords(lb1gt_1,235,283,514).  visible(lb1gt_1,514). happensAt(walking(lb1gt_1),514). orientation(lb1gt_2,6,514). coords(lb1gt_2,208,246,514).  visible(lb1gt_2,514). happensAt(walking(lb1gt_2),514).
orientation(lb1gt_1,84,515). coords(lb1gt_1,235,283,515).  visible(lb1gt_1,515). happensAt(walking(lb1gt_1),515). orientation(lb1gt_2,6,515). coords(lb1gt_2,208,246,515).  visible(lb1gt_2,515). happensAt(walking(lb1gt_2),515).
orientation(lb1gt_1,84,516). coords(lb1gt_1,235,283,516).  happensAt(disappear(lb1gt_1),516). happensAt(walking(lb1gt_1),516). orientation(lb1gt_2,6,516). coords(lb1gt_2,209,247,516).  visible(lb1gt_2,516). happensAt(walking(lb1gt_2),516).
#show holdsAt/2.