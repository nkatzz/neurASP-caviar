
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

	orientation(fra2gt_0,150,84). coords(fra2gt_0,173,132,84).  visible(fra2gt_0,84). happensAt(walking(fra2gt_0),84). orientation(fra2gt_5,150,84). coords(fra2gt_5,32,116,84).  visible(fra2gt_5,84). happensAt(walking(fra2gt_5),84).
orientation(fra2gt_0,150,85). coords(fra2gt_0,175,132,85).  visible(fra2gt_0,85). happensAt(walking(fra2gt_0),85). orientation(fra2gt_5,150,85). coords(fra2gt_5,31,116,85).  visible(fra2gt_5,85). happensAt(walking(fra2gt_5),85).
orientation(fra2gt_0,150,86). coords(fra2gt_0,178,133,86).  visible(fra2gt_0,86). happensAt(walking(fra2gt_0),86). orientation(fra2gt_5,150,86). coords(fra2gt_5,31,116,86).  visible(fra2gt_5,86). happensAt(walking(fra2gt_5),86).
orientation(fra2gt_0,150,87). coords(fra2gt_0,180,133,87).  visible(fra2gt_0,87). happensAt(walking(fra2gt_0),87). orientation(fra2gt_5,150,87). coords(fra2gt_5,31,116,87).  visible(fra2gt_5,87). happensAt(walking(fra2gt_5),87).
orientation(fra2gt_0,150,88). coords(fra2gt_0,181,134,88).  visible(fra2gt_0,88). happensAt(walking(fra2gt_0),88). orientation(fra2gt_5,150,88). coords(fra2gt_5,31,117,88).  visible(fra2gt_5,88). happensAt(walking(fra2gt_5),88).
orientation(fra2gt_0,150,89). coords(fra2gt_0,182,135,89).  visible(fra2gt_0,89). happensAt(walking(fra2gt_0),89). orientation(fra2gt_5,150,89). coords(fra2gt_5,31,117,89).  visible(fra2gt_5,89). happensAt(walking(fra2gt_5),89).
orientation(fra2gt_0,150,90). coords(fra2gt_0,183,135,90).  visible(fra2gt_0,90). happensAt(walking(fra2gt_0),90). orientation(fra2gt_5,150,90). coords(fra2gt_5,30,117,90).  visible(fra2gt_5,90). happensAt(walking(fra2gt_5),90).
orientation(fra2gt_0,150,91). coords(fra2gt_0,183,137,91).  visible(fra2gt_0,91). happensAt(walking(fra2gt_0),91). orientation(fra2gt_5,150,91). coords(fra2gt_5,30,117,91).  visible(fra2gt_5,91). happensAt(walking(fra2gt_5),91).
orientation(fra2gt_0,150,92). coords(fra2gt_0,184,140,92).  visible(fra2gt_0,92). happensAt(walking(fra2gt_0),92). orientation(fra2gt_5,150,92). coords(fra2gt_5,30,117,92).  visible(fra2gt_5,92). happensAt(walking(fra2gt_5),92).
orientation(fra2gt_0,150,93). coords(fra2gt_0,188,145,93).  visible(fra2gt_0,93). happensAt(walking(fra2gt_0),93). orientation(fra2gt_5,150,93). coords(fra2gt_5,29,119,93).  visible(fra2gt_5,93). happensAt(walking(fra2gt_5),93).
orientation(fra2gt_0,150,94). coords(fra2gt_0,191,150,94).  visible(fra2gt_0,94). happensAt(walking(fra2gt_0),94). orientation(fra2gt_5,150,94). coords(fra2gt_5,29,120,94).  visible(fra2gt_5,94). happensAt(walking(fra2gt_5),94).
orientation(fra2gt_0,150,95). coords(fra2gt_0,194,155,95).  visible(fra2gt_0,95). happensAt(walking(fra2gt_0),95). orientation(fra2gt_5,150,95). coords(fra2gt_5,29,120,95).  visible(fra2gt_5,95). happensAt(walking(fra2gt_5),95).
orientation(fra2gt_0,150,96). coords(fra2gt_0,196,159,96).  visible(fra2gt_0,96). happensAt(walking(fra2gt_0),96). orientation(fra2gt_5,150,96). coords(fra2gt_5,30,121,96).  visible(fra2gt_5,96). happensAt(walking(fra2gt_5),96).
orientation(fra2gt_0,150,97). coords(fra2gt_0,197,162,97).  visible(fra2gt_0,97). happensAt(walking(fra2gt_0),97). orientation(fra2gt_5,150,97). coords(fra2gt_5,30,122,97).  visible(fra2gt_5,97). happensAt(walking(fra2gt_5),97).
orientation(fra2gt_0,150,98). coords(fra2gt_0,198,164,98).  visible(fra2gt_0,98). happensAt(walking(fra2gt_0),98). orientation(fra2gt_5,150,98). coords(fra2gt_5,30,122,98).  visible(fra2gt_5,98). happensAt(walking(fra2gt_5),98).
orientation(fra2gt_0,150,99). coords(fra2gt_0,199,166,99).  visible(fra2gt_0,99). happensAt(walking(fra2gt_0),99). orientation(fra2gt_5,150,99). coords(fra2gt_5,29,123,99).  visible(fra2gt_5,99). happensAt(walking(fra2gt_5),99).
orientation(fra2gt_0,150,100). coords(fra2gt_0,200,168,100).  visible(fra2gt_0,100). happensAt(walking(fra2gt_0),100). orientation(fra2gt_5,150,100). coords(fra2gt_5,29,123,100).  visible(fra2gt_5,100). happensAt(walking(fra2gt_5),100).
orientation(fra2gt_0,150,101). coords(fra2gt_0,202,169,101).  visible(fra2gt_0,101). happensAt(walking(fra2gt_0),101). orientation(fra2gt_5,150,101). coords(fra2gt_5,29,123,101).  visible(fra2gt_5,101). happensAt(walking(fra2gt_5),101).
orientation(fra2gt_0,150,102). coords(fra2gt_0,203,170,102).  visible(fra2gt_0,102). happensAt(walking(fra2gt_0),102). orientation(fra2gt_5,150,102). coords(fra2gt_5,29,123,102).  visible(fra2gt_5,102). happensAt(walking(fra2gt_5),102).
orientation(fra2gt_0,150,103). coords(fra2gt_0,203,171,103).  visible(fra2gt_0,103). happensAt(walking(fra2gt_0),103). orientation(fra2gt_5,150,103). coords(fra2gt_5,29,123,103).  visible(fra2gt_5,103). happensAt(walking(fra2gt_5),103).
orientation(fra2gt_0,152,104). coords(fra2gt_0,204,173,104).  visible(fra2gt_0,104). happensAt(walking(fra2gt_0),104). orientation(fra2gt_5,150,104). coords(fra2gt_5,29,123,104).  visible(fra2gt_5,104). happensAt(walking(fra2gt_5),104).
orientation(fra2gt_0,166,105). coords(fra2gt_0,205,175,105).  visible(fra2gt_0,105). happensAt(walking(fra2gt_0),105). orientation(fra2gt_5,150,105). coords(fra2gt_5,29,123,105).  visible(fra2gt_5,105). happensAt(walking(fra2gt_5),105).
orientation(fra2gt_0,165,106). coords(fra2gt_0,208,177,106).  visible(fra2gt_0,106). happensAt(walking(fra2gt_0),106). orientation(fra2gt_5,150,106). coords(fra2gt_5,29,123,106).  visible(fra2gt_5,106). happensAt(walking(fra2gt_5),106).
orientation(fra2gt_0,168,107). coords(fra2gt_0,211,179,107).  visible(fra2gt_0,107). happensAt(walking(fra2gt_0),107). orientation(fra2gt_5,150,107). coords(fra2gt_5,29,124,107).  visible(fra2gt_5,107). happensAt(walking(fra2gt_5),107).
orientation(fra2gt_0,168,108). coords(fra2gt_0,215,181,108).  visible(fra2gt_0,108). happensAt(walking(fra2gt_0),108). orientation(fra2gt_5,150,108). coords(fra2gt_5,29,124,108).  visible(fra2gt_5,108). happensAt(walking(fra2gt_5),108).
#show holdsAt/2.