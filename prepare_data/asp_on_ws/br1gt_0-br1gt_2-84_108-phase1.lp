
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

	orientation(br1gt_0,157,84). coords(br1gt_0,183,141,84).  visible(br1gt_0,84). happensAt(inactive(br1gt_0),84). orientation(br1gt_2,171,84). coords(br1gt_2,86,207,84).  visible(br1gt_2,84). happensAt(inactive(br1gt_2),84).
orientation(br1gt_0,157,85). coords(br1gt_0,183,141,85).  visible(br1gt_0,85). happensAt(inactive(br1gt_0),85). orientation(br1gt_2,171,85). coords(br1gt_2,86,207,85).  visible(br1gt_2,85). happensAt(inactive(br1gt_2),85).
orientation(br1gt_0,157,86). coords(br1gt_0,183,141,86).  visible(br1gt_0,86). happensAt(inactive(br1gt_0),86). orientation(br1gt_2,171,86). coords(br1gt_2,86,207,86).  visible(br1gt_2,86). happensAt(inactive(br1gt_2),86).
orientation(br1gt_0,157,87). coords(br1gt_0,183,141,87).  visible(br1gt_0,87). happensAt(inactive(br1gt_0),87). orientation(br1gt_2,171,87). coords(br1gt_2,86,207,87).  visible(br1gt_2,87). happensAt(inactive(br1gt_2),87).
orientation(br1gt_0,157,88). coords(br1gt_0,183,141,88).  visible(br1gt_0,88). happensAt(inactive(br1gt_0),88). orientation(br1gt_2,171,88). coords(br1gt_2,86,207,88).  visible(br1gt_2,88). happensAt(inactive(br1gt_2),88).
orientation(br1gt_0,157,89). coords(br1gt_0,183,141,89).  visible(br1gt_0,89). happensAt(inactive(br1gt_0),89). orientation(br1gt_2,171,89). coords(br1gt_2,86,207,89).  visible(br1gt_2,89). happensAt(inactive(br1gt_2),89).
orientation(br1gt_0,157,90). coords(br1gt_0,183,141,90).  visible(br1gt_0,90). happensAt(inactive(br1gt_0),90). orientation(br1gt_2,171,90). coords(br1gt_2,86,207,90).  visible(br1gt_2,90). happensAt(inactive(br1gt_2),90).
orientation(br1gt_0,157,91). coords(br1gt_0,183,141,91).  visible(br1gt_0,91). happensAt(inactive(br1gt_0),91). orientation(br1gt_2,171,91). coords(br1gt_2,86,207,91).  visible(br1gt_2,91). happensAt(inactive(br1gt_2),91).
orientation(br1gt_0,157,92). coords(br1gt_0,183,141,92).  visible(br1gt_0,92). happensAt(inactive(br1gt_0),92). orientation(br1gt_2,171,92). coords(br1gt_2,86,207,92).  visible(br1gt_2,92). happensAt(inactive(br1gt_2),92).
orientation(br1gt_0,157,93). coords(br1gt_0,183,140,93).  visible(br1gt_0,93). happensAt(inactive(br1gt_0),93). orientation(br1gt_2,171,93). coords(br1gt_2,86,207,93).  visible(br1gt_2,93). happensAt(inactive(br1gt_2),93).
orientation(br1gt_0,157,94). coords(br1gt_0,183,140,94).  visible(br1gt_0,94). happensAt(inactive(br1gt_0),94). orientation(br1gt_2,171,94). coords(br1gt_2,86,207,94).  visible(br1gt_2,94). happensAt(inactive(br1gt_2),94).
orientation(br1gt_0,161,95). coords(br1gt_0,183,140,95).  visible(br1gt_0,95). happensAt(inactive(br1gt_0),95). orientation(br1gt_2,171,95). coords(br1gt_2,86,207,95).  visible(br1gt_2,95). happensAt(inactive(br1gt_2),95).
orientation(br1gt_0,159,96). coords(br1gt_0,183,140,96).  visible(br1gt_0,96). happensAt(inactive(br1gt_0),96). orientation(br1gt_2,171,96). coords(br1gt_2,86,207,96).  visible(br1gt_2,96). happensAt(inactive(br1gt_2),96).
orientation(br1gt_0,159,97). coords(br1gt_0,183,140,97).  visible(br1gt_0,97). happensAt(inactive(br1gt_0),97). orientation(br1gt_2,171,97). coords(br1gt_2,86,207,97).  visible(br1gt_2,97). happensAt(inactive(br1gt_2),97).
orientation(br1gt_0,159,98). coords(br1gt_0,183,140,98).  visible(br1gt_0,98). happensAt(inactive(br1gt_0),98). orientation(br1gt_2,171,98). coords(br1gt_2,86,207,98).  visible(br1gt_2,98). happensAt(inactive(br1gt_2),98).
orientation(br1gt_0,159,99). coords(br1gt_0,183,140,99).  visible(br1gt_0,99). happensAt(inactive(br1gt_0),99). orientation(br1gt_2,171,99). coords(br1gt_2,86,207,99).  visible(br1gt_2,99). happensAt(inactive(br1gt_2),99).
orientation(br1gt_0,159,100). coords(br1gt_0,183,140,100).  visible(br1gt_0,100). happensAt(inactive(br1gt_0),100). orientation(br1gt_2,171,100). coords(br1gt_2,86,207,100).  visible(br1gt_2,100). happensAt(inactive(br1gt_2),100).
orientation(br1gt_0,159,101). coords(br1gt_0,182,140,101).  visible(br1gt_0,101). happensAt(inactive(br1gt_0),101). orientation(br1gt_2,171,101). coords(br1gt_2,86,207,101).  visible(br1gt_2,101). happensAt(inactive(br1gt_2),101).
orientation(br1gt_0,159,102). coords(br1gt_0,182,140,102).  visible(br1gt_0,102). happensAt(inactive(br1gt_0),102). orientation(br1gt_2,171,102). coords(br1gt_2,86,207,102).  visible(br1gt_2,102). happensAt(inactive(br1gt_2),102).
orientation(br1gt_0,159,103). coords(br1gt_0,182,140,103).  visible(br1gt_0,103). happensAt(inactive(br1gt_0),103). orientation(br1gt_2,171,103). coords(br1gt_2,86,207,103).  visible(br1gt_2,103). happensAt(inactive(br1gt_2),103).
orientation(br1gt_0,159,104). coords(br1gt_0,182,140,104).  visible(br1gt_0,104). happensAt(inactive(br1gt_0),104). orientation(br1gt_2,171,104). coords(br1gt_2,86,207,104).  visible(br1gt_2,104). happensAt(inactive(br1gt_2),104).
orientation(br1gt_0,155,105). coords(br1gt_0,182,140,105).  visible(br1gt_0,105). happensAt(inactive(br1gt_0),105). orientation(br1gt_2,171,105). coords(br1gt_2,86,207,105).  visible(br1gt_2,105). happensAt(inactive(br1gt_2),105).
orientation(br1gt_0,159,106). coords(br1gt_0,183,141,106).  visible(br1gt_0,106). happensAt(inactive(br1gt_0),106). orientation(br1gt_2,171,106). coords(br1gt_2,86,207,106).  visible(br1gt_2,106). happensAt(inactive(br1gt_2),106).
orientation(br1gt_0,159,107). coords(br1gt_0,183,141,107).  visible(br1gt_0,107). happensAt(inactive(br1gt_0),107). orientation(br1gt_2,171,107). coords(br1gt_2,86,207,107).  visible(br1gt_2,107). happensAt(inactive(br1gt_2),107).
orientation(br1gt_0,159,108). coords(br1gt_0,183,141,108).  visible(br1gt_0,108). happensAt(inactive(br1gt_0),108). orientation(br1gt_2,171,108). coords(br1gt_2,86,207,108).  visible(br1gt_2,108). happensAt(inactive(br1gt_2),108).
#show holdsAt/2.