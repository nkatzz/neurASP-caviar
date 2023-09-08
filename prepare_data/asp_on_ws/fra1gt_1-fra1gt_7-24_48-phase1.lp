
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

	orientation(fra1gt_1,106,24). coords(fra1gt_1,278,12,24).  visible(fra1gt_1,24). happensAt(walking(fra1gt_1),24). orientation(fra1gt_7,136,24). coords(fra1gt_7,144,40,24).  visible(fra1gt_7,24). happensAt(walking(fra1gt_7),24).
orientation(fra1gt_1,106,25). coords(fra1gt_1,278,12,25).  visible(fra1gt_1,25). happensAt(walking(fra1gt_1),25). orientation(fra1gt_7,136,25). coords(fra1gt_7,144,40,25).  visible(fra1gt_7,25). happensAt(walking(fra1gt_7),25).
orientation(fra1gt_1,106,26). coords(fra1gt_1,278,11,26).  visible(fra1gt_1,26). happensAt(walking(fra1gt_1),26). orientation(fra1gt_7,136,26). coords(fra1gt_7,144,40,26).  visible(fra1gt_7,26). happensAt(walking(fra1gt_7),26).
orientation(fra1gt_1,106,27). coords(fra1gt_1,278,11,27).  visible(fra1gt_1,27). happensAt(walking(fra1gt_1),27). orientation(fra1gt_7,136,27). coords(fra1gt_7,144,40,27).  visible(fra1gt_7,27). happensAt(walking(fra1gt_7),27).
orientation(fra1gt_1,106,28). coords(fra1gt_1,278,11,28).  visible(fra1gt_1,28). happensAt(walking(fra1gt_1),28). orientation(fra1gt_7,136,28). coords(fra1gt_7,144,40,28).  visible(fra1gt_7,28). happensAt(walking(fra1gt_7),28).
orientation(fra1gt_1,106,29). coords(fra1gt_1,278,11,29).  visible(fra1gt_1,29). happensAt(walking(fra1gt_1),29). orientation(fra1gt_7,136,29). coords(fra1gt_7,144,40,29).  visible(fra1gt_7,29). happensAt(walking(fra1gt_7),29).
orientation(fra1gt_1,106,30). coords(fra1gt_1,278,11,30).  visible(fra1gt_1,30). happensAt(walking(fra1gt_1),30). orientation(fra1gt_7,136,30). coords(fra1gt_7,144,40,30).  visible(fra1gt_7,30). happensAt(walking(fra1gt_7),30).
orientation(fra1gt_1,106,31). coords(fra1gt_1,278,11,31).  visible(fra1gt_1,31). happensAt(walking(fra1gt_1),31). orientation(fra1gt_7,136,31). coords(fra1gt_7,144,40,31).  visible(fra1gt_7,31). happensAt(walking(fra1gt_7),31).
orientation(fra1gt_1,103,32). coords(fra1gt_1,278,11,32).  visible(fra1gt_1,32). happensAt(walking(fra1gt_1),32). orientation(fra1gt_7,136,32). coords(fra1gt_7,144,40,32).  visible(fra1gt_7,32). happensAt(walking(fra1gt_7),32).
orientation(fra1gt_1,103,33). coords(fra1gt_1,279,10,33).  visible(fra1gt_1,33). happensAt(walking(fra1gt_1),33). orientation(fra1gt_7,136,33). coords(fra1gt_7,144,40,33).  visible(fra1gt_7,33). happensAt(walking(fra1gt_7),33).
orientation(fra1gt_1,103,34). coords(fra1gt_1,279,8,34).  visible(fra1gt_1,34). happensAt(walking(fra1gt_1),34). orientation(fra1gt_7,136,34). coords(fra1gt_7,144,40,34).  visible(fra1gt_7,34). happensAt(walking(fra1gt_7),34).
orientation(fra1gt_1,103,35). coords(fra1gt_1,279,6,35).  visible(fra1gt_1,35). happensAt(walking(fra1gt_1),35). orientation(fra1gt_7,136,35). coords(fra1gt_7,145,40,35).  visible(fra1gt_7,35). happensAt(walking(fra1gt_7),35).
orientation(fra1gt_1,103,36). coords(fra1gt_1,281,6,36).  visible(fra1gt_1,36). happensAt(walking(fra1gt_1),36). orientation(fra1gt_7,136,36). coords(fra1gt_7,145,40,36).  visible(fra1gt_7,36). happensAt(walking(fra1gt_7),36).
orientation(fra1gt_1,103,37). coords(fra1gt_1,281,6,37).  visible(fra1gt_1,37). happensAt(walking(fra1gt_1),37). orientation(fra1gt_7,136,37). coords(fra1gt_7,145,40,37).  visible(fra1gt_7,37). happensAt(walking(fra1gt_7),37).
orientation(fra1gt_1,103,38). coords(fra1gt_1,281,5,38).  visible(fra1gt_1,38). happensAt(walking(fra1gt_1),38). orientation(fra1gt_7,136,38). coords(fra1gt_7,145,40,38).  visible(fra1gt_7,38). happensAt(walking(fra1gt_7),38).
orientation(fra1gt_1,103,39). coords(fra1gt_1,281,5,39).  visible(fra1gt_1,39). happensAt(walking(fra1gt_1),39). orientation(fra1gt_7,136,39). coords(fra1gt_7,145,40,39).  visible(fra1gt_7,39). happensAt(inactive(fra1gt_7),39).
orientation(fra1gt_1,103,40). coords(fra1gt_1,282,5,40).  visible(fra1gt_1,40). happensAt(walking(fra1gt_1),40). orientation(fra1gt_7,136,40). coords(fra1gt_7,145,40,40).  visible(fra1gt_7,40). happensAt(inactive(fra1gt_7),40).
orientation(fra1gt_1,103,41). coords(fra1gt_1,282,5,41).  visible(fra1gt_1,41). happensAt(walking(fra1gt_1),41). orientation(fra1gt_7,136,41). coords(fra1gt_7,145,40,41).  visible(fra1gt_7,41). happensAt(inactive(fra1gt_7),41).
orientation(fra1gt_1,103,42). coords(fra1gt_1,282,5,42).  visible(fra1gt_1,42). happensAt(walking(fra1gt_1),42). orientation(fra1gt_7,136,42). coords(fra1gt_7,145,40,42).  visible(fra1gt_7,42). happensAt(inactive(fra1gt_7),42).
orientation(fra1gt_1,103,43). coords(fra1gt_1,282,5,43).  visible(fra1gt_1,43). happensAt(walking(fra1gt_1),43). orientation(fra1gt_7,136,43). coords(fra1gt_7,145,40,43).  visible(fra1gt_7,43). happensAt(inactive(fra1gt_7),43).
orientation(fra1gt_1,103,44). coords(fra1gt_1,282,5,44).  visible(fra1gt_1,44). happensAt(walking(fra1gt_1),44). orientation(fra1gt_7,136,44). coords(fra1gt_7,145,40,44).  visible(fra1gt_7,44). happensAt(inactive(fra1gt_7),44).
orientation(fra1gt_1,103,45). coords(fra1gt_1,282,5,45).  visible(fra1gt_1,45). happensAt(walking(fra1gt_1),45). orientation(fra1gt_7,136,45). coords(fra1gt_7,145,40,45).  visible(fra1gt_7,45). happensAt(inactive(fra1gt_7),45).
orientation(fra1gt_1,103,46). coords(fra1gt_1,282,4,46).  visible(fra1gt_1,46). happensAt(walking(fra1gt_1),46). orientation(fra1gt_7,136,46). coords(fra1gt_7,144,40,46).  visible(fra1gt_7,46). happensAt(inactive(fra1gt_7),46).
orientation(fra1gt_1,103,47). coords(fra1gt_1,282,4,47).  visible(fra1gt_1,47). happensAt(walking(fra1gt_1),47). orientation(fra1gt_7,136,47). coords(fra1gt_7,144,40,47).  visible(fra1gt_7,47). happensAt(inactive(fra1gt_7),47).
orientation(fra1gt_1,103,48). coords(fra1gt_1,282,3,48).  visible(fra1gt_1,48). happensAt(walking(fra1gt_1),48). orientation(fra1gt_7,136,48). coords(fra1gt_7,144,41,48).  visible(fra1gt_7,48). happensAt(inactive(fra1gt_7),48).
#show holdsAt/2.