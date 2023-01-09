
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

	orientation(mc1gt_2,163,288). coords(mc1gt_2,187,178,288).  visible(mc1gt_2,288). happensAt(walking(mc1gt_2),288). orientation(mc1gt_3,135,288). coords(mc1gt_3,263,146,288).  visible(mc1gt_3,288). happensAt(walking(mc1gt_3),288).
orientation(mc1gt_2,163,289). coords(mc1gt_2,188,179,289).  visible(mc1gt_2,289). happensAt(walking(mc1gt_2),289). orientation(mc1gt_3,135,289). coords(mc1gt_3,264,148,289).  visible(mc1gt_3,289). happensAt(walking(mc1gt_3),289).
orientation(mc1gt_2,163,290). coords(mc1gt_2,190,180,290).  visible(mc1gt_2,290). happensAt(walking(mc1gt_2),290). orientation(mc1gt_3,135,290). coords(mc1gt_3,264,150,290).  visible(mc1gt_3,290). happensAt(walking(mc1gt_3),290).
orientation(mc1gt_2,166,291). coords(mc1gt_2,191,182,291).  visible(mc1gt_2,291). happensAt(walking(mc1gt_2),291). orientation(mc1gt_3,135,291). coords(mc1gt_3,265,153,291).  visible(mc1gt_3,291). happensAt(walking(mc1gt_3),291).
orientation(mc1gt_2,159,292). coords(mc1gt_2,191,182,292).  visible(mc1gt_2,292). happensAt(walking(mc1gt_2),292). orientation(mc1gt_3,135,292). coords(mc1gt_3,269,158,292).  visible(mc1gt_3,292). happensAt(walking(mc1gt_3),292).
orientation(mc1gt_2,160,293). coords(mc1gt_2,192,183,293).  visible(mc1gt_2,293). happensAt(walking(mc1gt_2),293). orientation(mc1gt_3,130,293). coords(mc1gt_3,271,162,293).  visible(mc1gt_3,293). happensAt(walking(mc1gt_3),293).
orientation(mc1gt_2,165,294). coords(mc1gt_2,192,184,294).  visible(mc1gt_2,294). happensAt(walking(mc1gt_2),294). orientation(mc1gt_3,125,294). coords(mc1gt_3,274,166,294).  visible(mc1gt_3,294). happensAt(walking(mc1gt_3),294).
orientation(mc1gt_2,167,295). coords(mc1gt_2,192,184,295).  visible(mc1gt_2,295). happensAt(walking(mc1gt_2),295). orientation(mc1gt_3,123,295). coords(mc1gt_3,276,169,295).  visible(mc1gt_3,295). happensAt(walking(mc1gt_3),295).
orientation(mc1gt_2,172,296). coords(mc1gt_2,193,185,296).  visible(mc1gt_2,296). happensAt(walking(mc1gt_2),296). orientation(mc1gt_3,124,296). coords(mc1gt_3,278,172,296).  visible(mc1gt_3,296). happensAt(walking(mc1gt_3),296).
orientation(mc1gt_2,168,297). coords(mc1gt_2,194,187,297).  visible(mc1gt_2,297). happensAt(walking(mc1gt_2),297). orientation(mc1gt_3,124,297). coords(mc1gt_3,280,174,297).  visible(mc1gt_3,297). happensAt(walking(mc1gt_3),297).
orientation(mc1gt_2,170,298). coords(mc1gt_2,196,188,298).  visible(mc1gt_2,298). happensAt(walking(mc1gt_2),298). orientation(mc1gt_3,131,298). coords(mc1gt_3,281,176,298).  visible(mc1gt_3,298). happensAt(walking(mc1gt_3),298).
orientation(mc1gt_2,172,299). coords(mc1gt_2,200,190,299).  visible(mc1gt_2,299). happensAt(walking(mc1gt_2),299). orientation(mc1gt_3,134,299). coords(mc1gt_3,283,176,299).  visible(mc1gt_3,299). happensAt(walking(mc1gt_3),299).
orientation(mc1gt_2,170,300). coords(mc1gt_2,202,192,300).  visible(mc1gt_2,300). happensAt(walking(mc1gt_2),300). orientation(mc1gt_3,135,300). coords(mc1gt_3,283,177,300).  visible(mc1gt_3,300). happensAt(walking(mc1gt_3),300).
orientation(mc1gt_2,173,301). coords(mc1gt_2,205,193,301).  visible(mc1gt_2,301). happensAt(walking(mc1gt_2),301). orientation(mc1gt_3,135,301). coords(mc1gt_3,284,179,301).  visible(mc1gt_3,301). happensAt(walking(mc1gt_3),301).
orientation(mc1gt_2,176,302). coords(mc1gt_2,207,195,302).  visible(mc1gt_2,302). happensAt(walking(mc1gt_2),302). orientation(mc1gt_3,135,302). coords(mc1gt_3,285,181,302).  visible(mc1gt_3,302). happensAt(walking(mc1gt_3),302).
orientation(mc1gt_2,178,303). coords(mc1gt_2,208,196,303).  visible(mc1gt_2,303). happensAt(walking(mc1gt_2),303). orientation(mc1gt_3,135,303). coords(mc1gt_3,286,182,303).  visible(mc1gt_3,303). happensAt(walking(mc1gt_3),303).
orientation(mc1gt_2,178,304). coords(mc1gt_2,211,197,304).  visible(mc1gt_2,304). happensAt(walking(mc1gt_2),304). orientation(mc1gt_3,135,304). coords(mc1gt_3,287,184,304).  visible(mc1gt_3,304). happensAt(walking(mc1gt_3),304).
orientation(mc1gt_2,178,305). coords(mc1gt_2,211,199,305).  visible(mc1gt_2,305). happensAt(walking(mc1gt_2),305). orientation(mc1gt_3,135,305). coords(mc1gt_3,287,185,305).  visible(mc1gt_3,305). happensAt(walking(mc1gt_3),305).
orientation(mc1gt_2,178,306). coords(mc1gt_2,212,200,306).  visible(mc1gt_2,306). happensAt(walking(mc1gt_2),306). orientation(mc1gt_3,152,306). coords(mc1gt_3,291,188,306).  visible(mc1gt_3,306). happensAt(walking(mc1gt_3),306).
orientation(mc1gt_2,178,307). coords(mc1gt_2,213,200,307).  visible(mc1gt_2,307). happensAt(walking(mc1gt_2),307). orientation(mc1gt_3,160,307). coords(mc1gt_3,293,192,307).  visible(mc1gt_3,307). happensAt(walking(mc1gt_3),307).
orientation(mc1gt_2,178,308). coords(mc1gt_2,213,202,308).  visible(mc1gt_2,308). happensAt(walking(mc1gt_2),308). orientation(mc1gt_3,172,308). coords(mc1gt_3,294,196,308).  visible(mc1gt_3,308). happensAt(walking(mc1gt_3),308).
orientation(mc1gt_2,178,309). coords(mc1gt_2,214,203,309).  visible(mc1gt_2,309). happensAt(walking(mc1gt_2),309). orientation(mc1gt_3,3,309). coords(mc1gt_3,297,199,309).  visible(mc1gt_3,309). happensAt(walking(mc1gt_3),309).
orientation(mc1gt_2,178,310). coords(mc1gt_2,214,204,310).  visible(mc1gt_2,310). happensAt(walking(mc1gt_2),310). orientation(mc1gt_3,1,310). coords(mc1gt_3,298,201,310).  visible(mc1gt_3,310). happensAt(walking(mc1gt_3),310).
orientation(mc1gt_2,178,311). coords(mc1gt_2,215,206,311).  visible(mc1gt_2,311). happensAt(walking(mc1gt_2),311). orientation(mc1gt_3,6,311). coords(mc1gt_3,299,203,311).  visible(mc1gt_3,311). happensAt(walking(mc1gt_3),311).
orientation(mc1gt_2,178,312). coords(mc1gt_2,217,208,312).  visible(mc1gt_2,312). happensAt(walking(mc1gt_2),312). orientation(mc1gt_3,178,312). coords(mc1gt_3,300,204,312).  visible(mc1gt_3,312). happensAt(walking(mc1gt_3),312).
#show holdsAt/2.