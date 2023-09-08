
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

	orientation(mc1gt_1,4,285). coords(mc1gt_1,222,215,285).  visible(mc1gt_1,285). happensAt(walking(mc1gt_1),285). orientation(mc1gt_2,152,285). coords(mc1gt_2,180,170,285).  visible(mc1gt_2,285). happensAt(walking(mc1gt_2),285).
orientation(mc1gt_1,7,286). coords(mc1gt_1,222,218,286).  visible(mc1gt_1,286). happensAt(walking(mc1gt_1),286). orientation(mc1gt_2,156,286). coords(mc1gt_2,183,173,286).  visible(mc1gt_2,286). happensAt(walking(mc1gt_2),286).
orientation(mc1gt_1,3,287). coords(mc1gt_1,222,221,287).  visible(mc1gt_1,287). happensAt(walking(mc1gt_1),287). orientation(mc1gt_2,165,287). coords(mc1gt_2,185,176,287).  visible(mc1gt_2,287). happensAt(walking(mc1gt_2),287).
orientation(mc1gt_1,1,288). coords(mc1gt_1,222,225,288).  visible(mc1gt_1,288). happensAt(walking(mc1gt_1),288). orientation(mc1gt_2,163,288). coords(mc1gt_2,187,178,288).  visible(mc1gt_2,288). happensAt(walking(mc1gt_2),288).
orientation(mc1gt_1,3,289). coords(mc1gt_1,222,228,289).  visible(mc1gt_1,289). happensAt(walking(mc1gt_1),289). orientation(mc1gt_2,163,289). coords(mc1gt_2,188,179,289).  visible(mc1gt_2,289). happensAt(walking(mc1gt_2),289).
orientation(mc1gt_1,0,290). coords(mc1gt_1,222,230,290).  visible(mc1gt_1,290). happensAt(walking(mc1gt_1),290). orientation(mc1gt_2,163,290). coords(mc1gt_2,190,180,290).  visible(mc1gt_2,290). happensAt(walking(mc1gt_2),290).
orientation(mc1gt_1,171,291). coords(mc1gt_1,222,232,291).  visible(mc1gt_1,291). happensAt(walking(mc1gt_1),291). orientation(mc1gt_2,166,291). coords(mc1gt_2,191,182,291).  visible(mc1gt_2,291). happensAt(walking(mc1gt_2),291).
orientation(mc1gt_1,172,292). coords(mc1gt_1,222,235,292).  visible(mc1gt_1,292). happensAt(walking(mc1gt_1),292). orientation(mc1gt_2,159,292). coords(mc1gt_2,191,182,292).  visible(mc1gt_2,292). happensAt(walking(mc1gt_2),292).
orientation(mc1gt_1,174,293). coords(mc1gt_1,222,237,293).  visible(mc1gt_1,293). happensAt(walking(mc1gt_1),293). orientation(mc1gt_2,160,293). coords(mc1gt_2,192,183,293).  visible(mc1gt_2,293). happensAt(walking(mc1gt_2),293).
orientation(mc1gt_1,0,294). coords(mc1gt_1,222,239,294).  visible(mc1gt_1,294). happensAt(walking(mc1gt_1),294). orientation(mc1gt_2,165,294). coords(mc1gt_2,192,184,294).  visible(mc1gt_2,294). happensAt(walking(mc1gt_2),294).
orientation(mc1gt_1,3,295). coords(mc1gt_1,222,239,295).  visible(mc1gt_1,295). happensAt(walking(mc1gt_1),295). orientation(mc1gt_2,167,295). coords(mc1gt_2,192,184,295).  visible(mc1gt_2,295). happensAt(walking(mc1gt_2),295).
orientation(mc1gt_1,178,296). coords(mc1gt_1,222,240,296).  visible(mc1gt_1,296). happensAt(walking(mc1gt_1),296). orientation(mc1gt_2,172,296). coords(mc1gt_2,193,185,296).  visible(mc1gt_2,296). happensAt(walking(mc1gt_2),296).
orientation(mc1gt_1,3,297). coords(mc1gt_1,222,240,297).  visible(mc1gt_1,297). happensAt(walking(mc1gt_1),297). orientation(mc1gt_2,168,297). coords(mc1gt_2,194,187,297).  visible(mc1gt_2,297). happensAt(walking(mc1gt_2),297).
orientation(mc1gt_1,6,298). coords(mc1gt_1,222,241,298).  visible(mc1gt_1,298). happensAt(walking(mc1gt_1),298). orientation(mc1gt_2,170,298). coords(mc1gt_2,196,188,298).  visible(mc1gt_2,298). happensAt(walking(mc1gt_2),298).
orientation(mc1gt_1,0,299). coords(mc1gt_1,222,243,299).  visible(mc1gt_1,299). happensAt(walking(mc1gt_1),299). orientation(mc1gt_2,172,299). coords(mc1gt_2,200,190,299).  visible(mc1gt_2,299). happensAt(walking(mc1gt_2),299).
orientation(mc1gt_1,1,300). coords(mc1gt_1,222,245,300).  visible(mc1gt_1,300). happensAt(walking(mc1gt_1),300). orientation(mc1gt_2,170,300). coords(mc1gt_2,202,192,300).  visible(mc1gt_2,300). happensAt(walking(mc1gt_2),300).
orientation(mc1gt_1,5,301). coords(mc1gt_1,223,248,301).  visible(mc1gt_1,301). happensAt(walking(mc1gt_1),301). orientation(mc1gt_2,173,301). coords(mc1gt_2,205,193,301).  visible(mc1gt_2,301). happensAt(walking(mc1gt_2),301).
orientation(mc1gt_1,14,302). coords(mc1gt_1,223,252,302).  visible(mc1gt_1,302). happensAt(walking(mc1gt_1),302). orientation(mc1gt_2,176,302). coords(mc1gt_2,207,195,302).  visible(mc1gt_2,302). happensAt(walking(mc1gt_2),302).
orientation(mc1gt_1,5,303). coords(mc1gt_1,223,255,303).  visible(mc1gt_1,303). happensAt(walking(mc1gt_1),303). orientation(mc1gt_2,178,303). coords(mc1gt_2,208,196,303).  visible(mc1gt_2,303). happensAt(walking(mc1gt_2),303).
orientation(mc1gt_1,7,304). coords(mc1gt_1,224,259,304).  visible(mc1gt_1,304). happensAt(walking(mc1gt_1),304). orientation(mc1gt_2,178,304). coords(mc1gt_2,211,197,304).  visible(mc1gt_2,304). happensAt(walking(mc1gt_2),304).
orientation(mc1gt_1,13,305). coords(mc1gt_1,224,262,305).  visible(mc1gt_1,305). happensAt(walking(mc1gt_1),305). orientation(mc1gt_2,178,305). coords(mc1gt_2,211,199,305).  visible(mc1gt_2,305). happensAt(walking(mc1gt_2),305).
orientation(mc1gt_1,12,306). coords(mc1gt_1,224,263,306).  visible(mc1gt_1,306). happensAt(walking(mc1gt_1),306). orientation(mc1gt_2,178,306). coords(mc1gt_2,212,200,306).  visible(mc1gt_2,306). happensAt(walking(mc1gt_2),306).
orientation(mc1gt_1,8,307). coords(mc1gt_1,225,264,307).  visible(mc1gt_1,307). happensAt(walking(mc1gt_1),307). orientation(mc1gt_2,178,307). coords(mc1gt_2,213,200,307).  visible(mc1gt_2,307). happensAt(walking(mc1gt_2),307).
orientation(mc1gt_1,15,308). coords(mc1gt_1,226,265,308).  visible(mc1gt_1,308). happensAt(walking(mc1gt_1),308). orientation(mc1gt_2,178,308). coords(mc1gt_2,213,202,308).  visible(mc1gt_2,308). happensAt(walking(mc1gt_2),308).
orientation(mc1gt_1,14,309). coords(mc1gt_1,226,267,309).  visible(mc1gt_1,309). happensAt(walking(mc1gt_1),309). orientation(mc1gt_2,178,309). coords(mc1gt_2,214,203,309).  visible(mc1gt_2,309). happensAt(walking(mc1gt_2),309).
#show holdsAt/2.