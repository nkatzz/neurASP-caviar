
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

	orientation(br1gt_1,167,24). coords(br1gt_1,165,212,24).  visible(br1gt_1,24). happensAt(walking(br1gt_1),24). orientation(br1gt_2,160,24). coords(br1gt_2,84,196,24).  visible(br1gt_2,24). happensAt(walking(br1gt_2),24).
orientation(br1gt_1,167,25). coords(br1gt_1,165,210,25).  visible(br1gt_1,25). happensAt(walking(br1gt_1),25). orientation(br1gt_2,160,25). coords(br1gt_2,83,196,25).  visible(br1gt_2,25). happensAt(walking(br1gt_2),25).
orientation(br1gt_1,167,26). coords(br1gt_1,163,210,26).  visible(br1gt_1,26). happensAt(walking(br1gt_1),26). orientation(br1gt_2,160,26). coords(br1gt_2,83,196,26).  visible(br1gt_2,26). happensAt(walking(br1gt_2),26).
orientation(br1gt_1,167,27). coords(br1gt_1,163,209,27).  visible(br1gt_1,27). happensAt(walking(br1gt_1),27). orientation(br1gt_2,160,27). coords(br1gt_2,83,196,27).  visible(br1gt_2,27). happensAt(walking(br1gt_2),27).
orientation(br1gt_1,167,28). coords(br1gt_1,162,208,28).  visible(br1gt_1,28). happensAt(walking(br1gt_1),28). orientation(br1gt_2,160,28). coords(br1gt_2,83,197,28).  visible(br1gt_2,28). happensAt(walking(br1gt_2),28).
orientation(br1gt_1,167,29). coords(br1gt_1,161,208,29).  visible(br1gt_1,29). happensAt(walking(br1gt_1),29). orientation(br1gt_2,160,29). coords(br1gt_2,84,199,29).  visible(br1gt_2,29). happensAt(walking(br1gt_2),29).
orientation(br1gt_1,167,30). coords(br1gt_1,160,207,30).  visible(br1gt_1,30). happensAt(walking(br1gt_1),30). orientation(br1gt_2,160,30). coords(br1gt_2,84,200,30).  visible(br1gt_2,30). happensAt(walking(br1gt_2),30).
orientation(br1gt_1,167,31). coords(br1gt_1,158,206,31).  visible(br1gt_1,31). happensAt(walking(br1gt_1),31). orientation(br1gt_2,160,31). coords(br1gt_2,83,197,31).  visible(br1gt_2,31). happensAt(walking(br1gt_2),31).
orientation(br1gt_1,167,32). coords(br1gt_1,156,206,32).  visible(br1gt_1,32). happensAt(walking(br1gt_1),32). orientation(br1gt_2,160,32). coords(br1gt_2,83,197,32).  visible(br1gt_2,32). happensAt(walking(br1gt_2),32).
orientation(br1gt_1,167,33). coords(br1gt_1,153,205,33).  visible(br1gt_1,33). happensAt(walking(br1gt_1),33). orientation(br1gt_2,160,33). coords(br1gt_2,84,199,33).  visible(br1gt_2,33). happensAt(walking(br1gt_2),33).
orientation(br1gt_1,167,34). coords(br1gt_1,150,205,34).  visible(br1gt_1,34). happensAt(walking(br1gt_1),34). orientation(br1gt_2,160,34). coords(br1gt_2,84,199,34).  visible(br1gt_2,34). happensAt(walking(br1gt_2),34).
orientation(br1gt_1,167,35). coords(br1gt_1,147,204,35).  visible(br1gt_1,35). happensAt(walking(br1gt_1),35). orientation(br1gt_2,160,35). coords(br1gt_2,84,200,35).  visible(br1gt_2,35). happensAt(walking(br1gt_2),35).
orientation(br1gt_1,167,36). coords(br1gt_1,144,202,36).  visible(br1gt_1,36). happensAt(walking(br1gt_1),36). orientation(br1gt_2,160,36). coords(br1gt_2,84,200,36).  visible(br1gt_2,36). happensAt(walking(br1gt_2),36).
orientation(br1gt_1,167,37). coords(br1gt_1,143,201,37).  visible(br1gt_1,37). happensAt(walking(br1gt_1),37). orientation(br1gt_2,169,37). coords(br1gt_2,85,201,37).  visible(br1gt_2,37). happensAt(walking(br1gt_2),37).
orientation(br1gt_1,167,38). coords(br1gt_1,142,201,38).  visible(br1gt_1,38). happensAt(walking(br1gt_1),38). orientation(br1gt_2,169,38). coords(br1gt_2,84,201,38).  visible(br1gt_2,38). happensAt(walking(br1gt_2),38).
orientation(br1gt_1,167,39). coords(br1gt_1,142,201,39).  visible(br1gt_1,39). happensAt(walking(br1gt_1),39). orientation(br1gt_2,169,39). coords(br1gt_2,85,202,39).  visible(br1gt_2,39). happensAt(walking(br1gt_2),39).
orientation(br1gt_1,167,40). coords(br1gt_1,141,200,40).  visible(br1gt_1,40). happensAt(walking(br1gt_1),40). orientation(br1gt_2,169,40). coords(br1gt_2,84,201,40).  visible(br1gt_2,40). happensAt(walking(br1gt_2),40).
orientation(br1gt_1,167,41). coords(br1gt_1,140,199,41).  visible(br1gt_1,41). happensAt(walking(br1gt_1),41). orientation(br1gt_2,169,41). coords(br1gt_2,85,204,41).  visible(br1gt_2,41). happensAt(walking(br1gt_2),41).
orientation(br1gt_1,167,42). coords(br1gt_1,139,197,42).  visible(br1gt_1,42). happensAt(walking(br1gt_1),42). orientation(br1gt_2,169,42). coords(br1gt_2,85,204,42).  visible(br1gt_2,42). happensAt(walking(br1gt_2),42).
orientation(br1gt_1,167,43). coords(br1gt_1,139,197,43).  visible(br1gt_1,43). happensAt(walking(br1gt_1),43). orientation(br1gt_2,169,43). coords(br1gt_2,86,204,43).  visible(br1gt_2,43). happensAt(walking(br1gt_2),43).
orientation(br1gt_1,167,44). coords(br1gt_1,135,197,44).  visible(br1gt_1,44). happensAt(walking(br1gt_1),44). orientation(br1gt_2,171,44). coords(br1gt_2,86,204,44).  visible(br1gt_2,44). happensAt(walking(br1gt_2),44).
orientation(br1gt_1,164,45). coords(br1gt_1,133,196,45).  visible(br1gt_1,45). happensAt(walking(br1gt_1),45). orientation(br1gt_2,171,45). coords(br1gt_2,85,204,45).  visible(br1gt_2,45). happensAt(walking(br1gt_2),45).
orientation(br1gt_1,164,46). coords(br1gt_1,130,195,46).  visible(br1gt_1,46). happensAt(walking(br1gt_1),46). orientation(br1gt_2,171,46). coords(br1gt_2,86,205,46).  visible(br1gt_2,46). happensAt(walking(br1gt_2),46).
orientation(br1gt_1,164,47). coords(br1gt_1,128,194,47).  visible(br1gt_1,47). happensAt(walking(br1gt_1),47). orientation(br1gt_2,171,47). coords(br1gt_2,86,205,47).  visible(br1gt_2,47). happensAt(walking(br1gt_2),47).
orientation(br1gt_1,164,48). coords(br1gt_1,125,193,48).  visible(br1gt_1,48). happensAt(walking(br1gt_1),48). orientation(br1gt_2,171,48). coords(br1gt_2,86,205,48).  visible(br1gt_2,48). happensAt(inactive(br1gt_2),48).
#show holdsAt/2.