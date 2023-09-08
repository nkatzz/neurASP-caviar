
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

	orientation(br1gt_1,164,84). coords(br1gt_1,87,171,84).  visible(br1gt_1,84). happensAt(walking(br1gt_1),84). orientation(br1gt_2,171,84). coords(br1gt_2,86,207,84).  visible(br1gt_2,84). happensAt(inactive(br1gt_2),84).
orientation(br1gt_1,164,85). coords(br1gt_1,85,171,85).  visible(br1gt_1,85). happensAt(walking(br1gt_1),85). orientation(br1gt_2,171,85). coords(br1gt_2,86,207,85).  visible(br1gt_2,85). happensAt(inactive(br1gt_2),85).
orientation(br1gt_1,164,86). coords(br1gt_1,84,171,86).  visible(br1gt_1,86). happensAt(walking(br1gt_1),86). orientation(br1gt_2,171,86). coords(br1gt_2,86,207,86).  visible(br1gt_2,86). happensAt(inactive(br1gt_2),86).
orientation(br1gt_1,164,87). coords(br1gt_1,84,171,87).  visible(br1gt_1,87). happensAt(walking(br1gt_1),87). orientation(br1gt_2,171,87). coords(br1gt_2,86,207,87).  visible(br1gt_2,87). happensAt(inactive(br1gt_2),87).
orientation(br1gt_1,164,88). coords(br1gt_1,83,170,88).  visible(br1gt_1,88). happensAt(walking(br1gt_1),88). orientation(br1gt_2,171,88). coords(br1gt_2,86,207,88).  visible(br1gt_2,88). happensAt(inactive(br1gt_2),88).
orientation(br1gt_1,164,89). coords(br1gt_1,83,170,89).  visible(br1gt_1,89). happensAt(walking(br1gt_1),89). orientation(br1gt_2,171,89). coords(br1gt_2,86,207,89).  visible(br1gt_2,89). happensAt(inactive(br1gt_2),89).
orientation(br1gt_1,164,90). coords(br1gt_1,83,170,90).  visible(br1gt_1,90). happensAt(walking(br1gt_1),90). orientation(br1gt_2,171,90). coords(br1gt_2,86,207,90).  visible(br1gt_2,90). happensAt(inactive(br1gt_2),90).
orientation(br1gt_1,164,91). coords(br1gt_1,83,170,91).  visible(br1gt_1,91). happensAt(walking(br1gt_1),91). orientation(br1gt_2,171,91). coords(br1gt_2,86,207,91).  visible(br1gt_2,91). happensAt(inactive(br1gt_2),91).
orientation(br1gt_1,164,92). coords(br1gt_1,83,170,92).  visible(br1gt_1,92). happensAt(walking(br1gt_1),92). orientation(br1gt_2,171,92). coords(br1gt_2,86,207,92).  visible(br1gt_2,92). happensAt(inactive(br1gt_2),92).
orientation(br1gt_1,164,93). coords(br1gt_1,81,170,93).  visible(br1gt_1,93). happensAt(walking(br1gt_1),93). orientation(br1gt_2,171,93). coords(br1gt_2,86,207,93).  visible(br1gt_2,93). happensAt(inactive(br1gt_2),93).
orientation(br1gt_1,164,94). coords(br1gt_1,80,169,94).  visible(br1gt_1,94). happensAt(walking(br1gt_1),94). orientation(br1gt_2,171,94). coords(br1gt_2,86,207,94).  visible(br1gt_2,94). happensAt(inactive(br1gt_2),94).
orientation(br1gt_1,164,95). coords(br1gt_1,78,169,95).  visible(br1gt_1,95). happensAt(walking(br1gt_1),95). orientation(br1gt_2,171,95). coords(br1gt_2,86,207,95).  visible(br1gt_2,95). happensAt(inactive(br1gt_2),95).
orientation(br1gt_1,164,96). coords(br1gt_1,77,168,96).  visible(br1gt_1,96). happensAt(walking(br1gt_1),96). orientation(br1gt_2,171,96). coords(br1gt_2,86,207,96).  visible(br1gt_2,96). happensAt(inactive(br1gt_2),96).
orientation(br1gt_1,164,97). coords(br1gt_1,75,167,97).  visible(br1gt_1,97). happensAt(walking(br1gt_1),97). orientation(br1gt_2,171,97). coords(br1gt_2,86,207,97).  visible(br1gt_2,97). happensAt(inactive(br1gt_2),97).
orientation(br1gt_1,164,98). coords(br1gt_1,75,167,98).  visible(br1gt_1,98). happensAt(walking(br1gt_1),98). orientation(br1gt_2,171,98). coords(br1gt_2,86,207,98).  visible(br1gt_2,98). happensAt(inactive(br1gt_2),98).
orientation(br1gt_1,164,99). coords(br1gt_1,75,167,99).  visible(br1gt_1,99). happensAt(walking(br1gt_1),99). orientation(br1gt_2,171,99). coords(br1gt_2,86,207,99).  visible(br1gt_2,99). happensAt(inactive(br1gt_2),99).
orientation(br1gt_1,164,100). coords(br1gt_1,75,166,100).  visible(br1gt_1,100). happensAt(walking(br1gt_1),100). orientation(br1gt_2,171,100). coords(br1gt_2,86,207,100).  visible(br1gt_2,100). happensAt(inactive(br1gt_2),100).
orientation(br1gt_1,164,101). coords(br1gt_1,75,166,101).  visible(br1gt_1,101). happensAt(walking(br1gt_1),101). orientation(br1gt_2,171,101). coords(br1gt_2,86,207,101).  visible(br1gt_2,101). happensAt(inactive(br1gt_2),101).
orientation(br1gt_1,164,102). coords(br1gt_1,75,166,102).  visible(br1gt_1,102). happensAt(walking(br1gt_1),102). orientation(br1gt_2,171,102). coords(br1gt_2,86,207,102).  visible(br1gt_2,102). happensAt(inactive(br1gt_2),102).
orientation(br1gt_1,164,103). coords(br1gt_1,74,166,103).  visible(br1gt_1,103). happensAt(walking(br1gt_1),103). orientation(br1gt_2,171,103). coords(br1gt_2,86,207,103).  visible(br1gt_2,103). happensAt(inactive(br1gt_2),103).
orientation(br1gt_1,164,104). coords(br1gt_1,73,164,104).  visible(br1gt_1,104). happensAt(walking(br1gt_1),104). orientation(br1gt_2,171,104). coords(br1gt_2,86,207,104).  visible(br1gt_2,104). happensAt(inactive(br1gt_2),104).
orientation(br1gt_1,164,105). coords(br1gt_1,73,164,105).  visible(br1gt_1,105). happensAt(walking(br1gt_1),105). orientation(br1gt_2,171,105). coords(br1gt_2,86,207,105).  visible(br1gt_2,105). happensAt(inactive(br1gt_2),105).
orientation(br1gt_1,164,106). coords(br1gt_1,71,164,106).  visible(br1gt_1,106). happensAt(walking(br1gt_1),106). orientation(br1gt_2,171,106). coords(br1gt_2,86,207,106).  visible(br1gt_2,106). happensAt(inactive(br1gt_2),106).
orientation(br1gt_1,164,107). coords(br1gt_1,71,164,107).  visible(br1gt_1,107). happensAt(walking(br1gt_1),107). orientation(br1gt_2,171,107). coords(br1gt_2,86,207,107).  visible(br1gt_2,107). happensAt(inactive(br1gt_2),107).
orientation(br1gt_1,164,108). coords(br1gt_1,69,165,108).  visible(br1gt_1,108). happensAt(walking(br1gt_1),108). orientation(br1gt_2,171,108). coords(br1gt_2,86,207,108).  visible(br1gt_2,108). happensAt(inactive(br1gt_2),108).
#show holdsAt/2.