
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

	orientation(bww1gt_1,0,494). coords(bww1gt_1,105,264,494).  visible(bww1gt_1,494). happensAt(inactive(bww1gt_1),494). orientation(bww1gt_2,154,494). coords(bww1gt_2,179,178,494).  visible(bww1gt_2,494). happensAt(walking(bww1gt_2),494).
orientation(bww1gt_1,0,495). coords(bww1gt_1,105,264,495).  visible(bww1gt_1,495). happensAt(inactive(bww1gt_1),495). orientation(bww1gt_2,156,495). coords(bww1gt_2,180,179,495).  visible(bww1gt_2,495). happensAt(walking(bww1gt_2),495).
orientation(bww1gt_1,0,496). coords(bww1gt_1,105,264,496).  visible(bww1gt_1,496). happensAt(inactive(bww1gt_1),496). orientation(bww1gt_2,150,496). coords(bww1gt_2,180,180,496).  visible(bww1gt_2,496). happensAt(walking(bww1gt_2),496).
orientation(bww1gt_1,173,497). coords(bww1gt_1,104,263,497).  visible(bww1gt_1,497). happensAt(inactive(bww1gt_1),497). orientation(bww1gt_2,153,497). coords(bww1gt_2,180,180,497).  visible(bww1gt_2,497). happensAt(walking(bww1gt_2),497).
orientation(bww1gt_1,173,498). coords(bww1gt_1,104,263,498).  visible(bww1gt_1,498). happensAt(inactive(bww1gt_1),498). orientation(bww1gt_2,152,498). coords(bww1gt_2,181,181,498).  visible(bww1gt_2,498). happensAt(walking(bww1gt_2),498).
orientation(bww1gt_1,173,499). coords(bww1gt_1,104,263,499).  visible(bww1gt_1,499). happensAt(inactive(bww1gt_1),499). orientation(bww1gt_2,164,499). coords(bww1gt_2,181,182,499).  visible(bww1gt_2,499). happensAt(walking(bww1gt_2),499).
orientation(bww1gt_1,173,500). coords(bww1gt_1,104,263,500).  visible(bww1gt_1,500). happensAt(inactive(bww1gt_1),500). orientation(bww1gt_2,169,500). coords(bww1gt_2,181,183,500).  visible(bww1gt_2,500). happensAt(walking(bww1gt_2),500).
orientation(bww1gt_1,173,501). coords(bww1gt_1,104,263,501).  visible(bww1gt_1,501). happensAt(inactive(bww1gt_1),501). orientation(bww1gt_2,169,501). coords(bww1gt_2,181,183,501).  visible(bww1gt_2,501). happensAt(walking(bww1gt_2),501).
orientation(bww1gt_1,173,502). coords(bww1gt_1,104,263,502).  visible(bww1gt_1,502). happensAt(inactive(bww1gt_1),502). orientation(bww1gt_2,169,502). coords(bww1gt_2,181,184,502).  visible(bww1gt_2,502). happensAt(walking(bww1gt_2),502).
orientation(bww1gt_1,173,503). coords(bww1gt_1,104,263,503).  visible(bww1gt_1,503). happensAt(inactive(bww1gt_1),503). orientation(bww1gt_2,169,503). coords(bww1gt_2,181,186,503).  visible(bww1gt_2,503). happensAt(walking(bww1gt_2),503).
orientation(bww1gt_1,173,504). coords(bww1gt_1,103,263,504).  visible(bww1gt_1,504). happensAt(walking(bww1gt_1),504). orientation(bww1gt_2,169,504). coords(bww1gt_2,181,188,504).  visible(bww1gt_2,504). happensAt(walking(bww1gt_2),504).
orientation(bww1gt_1,173,505). coords(bww1gt_1,103,263,505).  visible(bww1gt_1,505). happensAt(walking(bww1gt_1),505). orientation(bww1gt_2,169,505). coords(bww1gt_2,181,189,505).  visible(bww1gt_2,505). happensAt(walking(bww1gt_2),505).
orientation(bww1gt_1,173,506). coords(bww1gt_1,103,263,506).  visible(bww1gt_1,506). happensAt(walking(bww1gt_1),506). orientation(bww1gt_2,169,506). coords(bww1gt_2,181,190,506).  visible(bww1gt_2,506). happensAt(walking(bww1gt_2),506).
orientation(bww1gt_1,173,507). coords(bww1gt_1,103,263,507).  visible(bww1gt_1,507). happensAt(walking(bww1gt_1),507). orientation(bww1gt_2,169,507). coords(bww1gt_2,181,192,507).  visible(bww1gt_2,507). happensAt(walking(bww1gt_2),507).
orientation(bww1gt_1,173,508). coords(bww1gt_1,103,263,508).  visible(bww1gt_1,508). happensAt(walking(bww1gt_1),508). orientation(bww1gt_2,169,508). coords(bww1gt_2,181,192,508).  visible(bww1gt_2,508). happensAt(walking(bww1gt_2),508).
orientation(bww1gt_1,173,509). coords(bww1gt_1,103,263,509).  visible(bww1gt_1,509). happensAt(walking(bww1gt_1),509). orientation(bww1gt_2,169,509). coords(bww1gt_2,181,193,509).  visible(bww1gt_2,509). happensAt(walking(bww1gt_2),509).
orientation(bww1gt_1,173,510). coords(bww1gt_1,102,263,510).  visible(bww1gt_1,510). happensAt(walking(bww1gt_1),510). orientation(bww1gt_2,166,510). coords(bww1gt_2,181,193,510).  visible(bww1gt_2,510). happensAt(walking(bww1gt_2),510).
orientation(bww1gt_1,173,511). coords(bww1gt_1,102,263,511).  visible(bww1gt_1,511). happensAt(walking(bww1gt_1),511). orientation(bww1gt_2,168,511). coords(bww1gt_2,181,194,511).  visible(bww1gt_2,511). happensAt(walking(bww1gt_2),511).
orientation(bww1gt_1,173,512). coords(bww1gt_1,102,263,512).  visible(bww1gt_1,512). happensAt(walking(bww1gt_1),512). orientation(bww1gt_2,168,512). coords(bww1gt_2,181,195,512).  visible(bww1gt_2,512). happensAt(walking(bww1gt_2),512).
orientation(bww1gt_1,173,513). coords(bww1gt_1,102,263,513).  visible(bww1gt_1,513). happensAt(walking(bww1gt_1),513). orientation(bww1gt_2,169,513). coords(bww1gt_2,181,195,513).  visible(bww1gt_2,513). happensAt(walking(bww1gt_2),513).
orientation(bww1gt_1,173,514). coords(bww1gt_1,102,263,514).  visible(bww1gt_1,514). happensAt(walking(bww1gt_1),514). orientation(bww1gt_2,169,514). coords(bww1gt_2,181,196,514).  visible(bww1gt_2,514). happensAt(walking(bww1gt_2),514).
orientation(bww1gt_1,173,515). coords(bww1gt_1,102,262,515).  visible(bww1gt_1,515). happensAt(walking(bww1gt_1),515). orientation(bww1gt_2,166,515). coords(bww1gt_2,181,196,515).  visible(bww1gt_2,515). happensAt(walking(bww1gt_2),515).
orientation(bww1gt_1,173,516). coords(bww1gt_1,102,262,516).  visible(bww1gt_1,516). happensAt(walking(bww1gt_1),516). orientation(bww1gt_2,169,516). coords(bww1gt_2,180,196,516).  visible(bww1gt_2,516). happensAt(walking(bww1gt_2),516).
orientation(bww1gt_1,173,517). coords(bww1gt_1,101,261,517).  visible(bww1gt_1,517). happensAt(walking(bww1gt_1),517). orientation(bww1gt_2,169,517). coords(bww1gt_2,180,197,517).  visible(bww1gt_2,517). happensAt(walking(bww1gt_2),517).
orientation(bww1gt_1,173,518). coords(bww1gt_1,101,261,518).  visible(bww1gt_1,518). happensAt(walking(bww1gt_1),518). orientation(bww1gt_2,169,518). coords(bww1gt_2,180,197,518).  visible(bww1gt_2,518). happensAt(walking(bww1gt_2),518).
#show holdsAt/2.