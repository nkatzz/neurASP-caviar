
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

	orientation(mws1gt_1,160,97). coords(mws1gt_1,91,177,97).  visible(mws1gt_1,97). happensAt(walking(mws1gt_1),97). orientation(mws1gt_2,162,97). coords(mws1gt_2,34,188,97).  visible(mws1gt_2,97). happensAt(walking(mws1gt_2),97).
orientation(mws1gt_1,160,98). coords(mws1gt_1,91,177,98).  visible(mws1gt_1,98). happensAt(walking(mws1gt_1),98). orientation(mws1gt_2,162,98). coords(mws1gt_2,34,189,98).  visible(mws1gt_2,98). happensAt(walking(mws1gt_2),98).
orientation(mws1gt_1,160,99). coords(mws1gt_1,91,177,99).  visible(mws1gt_1,99). happensAt(walking(mws1gt_1),99). orientation(mws1gt_2,162,99). coords(mws1gt_2,34,190,99).  visible(mws1gt_2,99). happensAt(walking(mws1gt_2),99).
orientation(mws1gt_1,155,100). coords(mws1gt_1,91,176,100).  visible(mws1gt_1,100). happensAt(walking(mws1gt_1),100). orientation(mws1gt_2,0,100). coords(mws1gt_2,34,189,100).  visible(mws1gt_2,100). happensAt(walking(mws1gt_2),100).
orientation(mws1gt_1,155,101). coords(mws1gt_1,91,176,101).  visible(mws1gt_1,101). happensAt(walking(mws1gt_1),101). orientation(mws1gt_2,0,101). coords(mws1gt_2,34,190,101).  visible(mws1gt_2,101). happensAt(walking(mws1gt_2),101).
orientation(mws1gt_1,155,102). coords(mws1gt_1,91,175,102).  visible(mws1gt_1,102). happensAt(walking(mws1gt_1),102). orientation(mws1gt_2,0,102). coords(mws1gt_2,34,191,102).  visible(mws1gt_2,102). happensAt(walking(mws1gt_2),102).
orientation(mws1gt_1,155,103). coords(mws1gt_1,94,174,103).  visible(mws1gt_1,103). happensAt(walking(mws1gt_1),103). orientation(mws1gt_2,167,103). coords(mws1gt_2,34,192,103).  visible(mws1gt_2,103). happensAt(walking(mws1gt_2),103).
orientation(mws1gt_1,155,104). coords(mws1gt_1,96,174,104).  visible(mws1gt_1,104). happensAt(walking(mws1gt_1),104). orientation(mws1gt_2,167,104). coords(mws1gt_2,34,193,104).  visible(mws1gt_2,104). happensAt(walking(mws1gt_2),104).
orientation(mws1gt_1,155,105). coords(mws1gt_1,98,173,105).  visible(mws1gt_1,105). happensAt(walking(mws1gt_1),105). orientation(mws1gt_2,167,105). coords(mws1gt_2,34,194,105).  visible(mws1gt_2,105). happensAt(walking(mws1gt_2),105).
orientation(mws1gt_1,155,106). coords(mws1gt_1,98,173,106).  visible(mws1gt_1,106). happensAt(walking(mws1gt_1),106). orientation(mws1gt_2,164,106). coords(mws1gt_2,34,195,106).  visible(mws1gt_2,106). happensAt(walking(mws1gt_2),106).
orientation(mws1gt_1,155,107). coords(mws1gt_1,99,173,107).  visible(mws1gt_1,107). happensAt(walking(mws1gt_1),107). orientation(mws1gt_2,164,107). coords(mws1gt_2,34,196,107).  visible(mws1gt_2,107). happensAt(walking(mws1gt_2),107).
orientation(mws1gt_1,155,108). coords(mws1gt_1,100,173,108).  visible(mws1gt_1,108). happensAt(walking(mws1gt_1),108). orientation(mws1gt_2,164,108). coords(mws1gt_2,34,197,108).  visible(mws1gt_2,108). happensAt(walking(mws1gt_2),108).
orientation(mws1gt_1,155,109). coords(mws1gt_1,102,172,109).  visible(mws1gt_1,109). happensAt(walking(mws1gt_1),109). orientation(mws1gt_2,164,109). coords(mws1gt_2,34,197,109).  visible(mws1gt_2,109). happensAt(walking(mws1gt_2),109).
orientation(mws1gt_1,155,110). coords(mws1gt_1,102,172,110).  visible(mws1gt_1,110). happensAt(walking(mws1gt_1),110). orientation(mws1gt_2,164,110). coords(mws1gt_2,34,197,110).  visible(mws1gt_2,110). happensAt(walking(mws1gt_2),110).
orientation(mws1gt_1,155,111). coords(mws1gt_1,102,172,111).  visible(mws1gt_1,111). happensAt(walking(mws1gt_1),111). orientation(mws1gt_2,164,111). coords(mws1gt_2,34,198,111).  visible(mws1gt_2,111). happensAt(walking(mws1gt_2),111).
orientation(mws1gt_1,155,112). coords(mws1gt_1,102,171,112).  visible(mws1gt_1,112). happensAt(walking(mws1gt_1),112). orientation(mws1gt_2,164,112). coords(mws1gt_2,34,198,112).  visible(mws1gt_2,112). happensAt(walking(mws1gt_2),112).
orientation(mws1gt_1,155,113). coords(mws1gt_1,102,171,113).  visible(mws1gt_1,113). happensAt(walking(mws1gt_1),113). orientation(mws1gt_2,164,113). coords(mws1gt_2,34,200,113).  visible(mws1gt_2,113). happensAt(walking(mws1gt_2),113).
orientation(mws1gt_1,155,114). coords(mws1gt_1,102,171,114).  visible(mws1gt_1,114). happensAt(walking(mws1gt_1),114). orientation(mws1gt_2,164,114). coords(mws1gt_2,34,200,114).  visible(mws1gt_2,114). happensAt(walking(mws1gt_2),114).
orientation(mws1gt_1,155,115). coords(mws1gt_1,102,170,115).  visible(mws1gt_1,115). happensAt(walking(mws1gt_1),115). orientation(mws1gt_2,164,115). coords(mws1gt_2,34,201,115).  visible(mws1gt_2,115). happensAt(walking(mws1gt_2),115).
orientation(mws1gt_1,155,116). coords(mws1gt_1,103,169,116).  visible(mws1gt_1,116). happensAt(walking(mws1gt_1),116). orientation(mws1gt_2,164,116). coords(mws1gt_2,34,202,116).  visible(mws1gt_2,116). happensAt(walking(mws1gt_2),116).
orientation(mws1gt_1,155,117). coords(mws1gt_1,105,169,117).  visible(mws1gt_1,117). happensAt(walking(mws1gt_1),117). orientation(mws1gt_2,164,117). coords(mws1gt_2,34,203,117).  visible(mws1gt_2,117). happensAt(walking(mws1gt_2),117).
orientation(mws1gt_1,155,118). coords(mws1gt_1,107,169,118).  visible(mws1gt_1,118). happensAt(walking(mws1gt_1),118). orientation(mws1gt_2,164,118). coords(mws1gt_2,34,203,118).  visible(mws1gt_2,118). happensAt(walking(mws1gt_2),118).
orientation(mws1gt_1,155,119). coords(mws1gt_1,108,168,119).  visible(mws1gt_1,119). happensAt(walking(mws1gt_1),119). orientation(mws1gt_2,164,119). coords(mws1gt_2,34,204,119).  visible(mws1gt_2,119). happensAt(walking(mws1gt_2),119).
orientation(mws1gt_1,155,120). coords(mws1gt_1,110,168,120).  visible(mws1gt_1,120). happensAt(walking(mws1gt_1),120). orientation(mws1gt_2,164,120). coords(mws1gt_2,34,205,120).  visible(mws1gt_2,120). happensAt(walking(mws1gt_2),120).
orientation(mws1gt_1,155,121). coords(mws1gt_1,111,167,121).  visible(mws1gt_1,121). happensAt(walking(mws1gt_1),121). orientation(mws1gt_2,164,121). coords(mws1gt_2,34,205,121).  visible(mws1gt_2,121). happensAt(walking(mws1gt_2),121).
#show holdsAt/2.